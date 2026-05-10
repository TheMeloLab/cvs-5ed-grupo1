import cv2
import numpy as np


class FramePreprocessor:
    """
    Pipeline de pré-processamento adaptativo para ANPR.

    Analisa cada frame automaticamente e decide quais as
    técnicas a aplicar consoante os problemas detetados:

        - Frame escuro/sobreexposto  → correção de gama automática
        - Baixo contraste            → CLAHE
        - Desfocado / motion blur    → sharpening
        - Ruído de sensor/compressão → denoising (sempre ativo)
        - Resolução fora do intervalo → resize

    Não requer nenhuma configuração manual por parte do utilizador.
    """

    def __init__(
        self,
        min_height: int = 480,
        max_height: int = 1080,
    ):
        self.min_height = min_height
        self.max_height = max_height

        # Criados uma vez e reutilizados em cada frame
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._clahe_plate = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica o pipeline adaptativo a um frame BGR.
        Analisa o frame e decide automaticamente o que aplicar.
        Devolve frame BGR melhorado.
        """
        if frame is None or frame.size == 0:
            return frame

        frame = self._resize(frame)

        diagnosis = self._analyze(frame)

        # Denoising: quase sempre útil — ruído piora todas as etapas seguintes
        frame = self._denoise(frame)

        # CLAHE: só se o contraste for baixo ou o frame estiver escuro
        if diagnosis["low_contrast"] or diagnosis["dark"]:
            frame = self._apply_clahe(frame)

        # Sharpening: só se o frame estiver desfocado
        if diagnosis["blurry"]:
            frame = self._sharpen(frame)

        # Correção de gama: só se a exposição estiver errada
        if diagnosis["dark"] or diagnosis["bright"]:
            frame = self._auto_gamma_correction(frame)

        # Normalização: sempre por último, estabiliza o histograma final
        frame = self._normalize(frame)

        return frame

    def process_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Pipeline mais agressivo aplicado ao recorte da matrícula.
        Devolve imagem binária em escala de cinzentos (ideal para Tesseract).
        """
        if crop is None or crop.size == 0:
            return crop

        # Tesseract funciona melhor com altura mínima de 64 px
        h, w = crop.shape[:2]
        if h < 64:
            scale = 64 / h
            crop = cv2.resize(
                crop, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self._clahe_plate.apply(gray)

        # Filtro bilateral: remove ruído mas preserva as bordas dos caracteres
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Threshold adaptativo: lida com iluminação não uniforme na matrícula
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # Fecho morfológico: fecha lacunas nos caracteres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    # ------------------------------------------------------------------
    # Análise automática
    # ------------------------------------------------------------------

    def _analyze(self, frame: np.ndarray) -> dict:
        """
        Analisa o frame e devolve um diagnóstico com os problemas detetados.

        Métricas usadas:
          - mean:      brilho médio (0-255)
          - std:       desvio padrão dos píxeis = contraste global
          - laplacian: variância do laplaciano = nitidez (quanto maior, mais nítido)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return {
            "dark":         mean < 60,
            "bright":       mean > 200,
            "low_contrast": std < 40,
            "blurry":       laplacian_var < 100,
        }

    # ------------------------------------------------------------------
    # Etapas internas
    # ------------------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if h < self.min_height:
            scale = self.min_height / h
            frame = cv2.resize(
                frame, (int(w * scale), self.min_height),
                interpolation=cv2.INTER_CUBIC
            )
        elif h > self.max_height:
            scale = self.max_height / h
            frame = cv2.resize(
                frame, (int(w * scale), self.max_height),
                interpolation=cv2.INTER_AREA
            )
        return frame

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            frame, None,
            h=10, hColor=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        # Aplicado só no canal L do espaço LAB para não distorcer as cores
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        # Unsharp masking: realça bordas subtraindo uma versão desfocada
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        return cv2.addWeighted(frame, 2.5, blurred, -1.5, 0)

    def _auto_gamma_correction(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        if mean_brightness < 0.01:
            return frame
        gamma = float(np.clip(np.log(0.5) / np.log(mean_brightness), 0.6, 1.8))
        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(frame, table)

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
