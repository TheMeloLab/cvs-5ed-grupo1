import re
import pytesseract


class PlateExtractor:
    """
    Extrai o texto de matrículas a partir de imagens pré-processadas.
    Devolve o texto da matrícula limpo, sem validação de formato específico.
    """

    def __init__(self):
        # Configuração do Tesseract:
        # --psm 8 → trata a imagem como uma única palavra
        # --oem 3 → usa o motor LSTM (mais preciso)
        # -c tessedit_char_whitelist → só aceita letras e números
        self._config = (
            "--psm 8 --oem 3 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )


    def extract(self, plate_image) -> str:

        # Devolve string vazia se não conseguir ler nada útil
        if plate_image is None or plate_image.size == 0:
            return ""

        raw_text = pytesseract.image_to_string(plate_image, config=self._config)
        cleaned = self._clean(raw_text)

        return cleaned

    # ------------------------------------------------------------------
    # Etapas internas
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        # Converte todas as letras para maiúsculas e remove espaços e newlines
        text = text.upper().strip()
        # Remove qualquer caractere que não seja letra ou número
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text
