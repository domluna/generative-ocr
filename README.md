# Generative OCR

For openai put your api key in .env

For gemini you have to setup the application through the google cloud sdk. You might be able to use an api key as well but in Canada we don't get access to such wonders.

The API for the ocr function is:

```python
def ocr(
    path_to_image: Path,
    questions: List[str],
    provider: str = "openai",
):
```

`test.py` usage:

```bash
NAME
    test.py

SYNOPSIS
    test.py <flags>

FLAGS
    --run_receipt_tests=RUN_RECEIPT_TESTS
        Default: False
    --run_truckticket_tests=RUN_TRUCKTICKET_TESTS
        Default: False
    --run_paystub_tests=RUN_PAYSTUB_TESTS
        Default: False
    --run_mortgage_tests=RUN_MORTGAGE_TESTS
        Default: False
    --run_insurance_tests=RUN_INSURANCE_TESTS
        Default: False
    --run_vaccination_tests=RUN_VACCINATION_TESTS
        Default: False
    -p, --provider=PROVIDER
        Default: 'openai'
```

`aws.py` calls AWS Textract on vaccination test set.
