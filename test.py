import os
import pathlib
from ocr import ocr, extract_and_format_numbers
from dataclasses import dataclass
from typing import List
from pathlib import Path
import fire

# Get the absolute path to the directory containing this script
script_dir = pathlib.Path(os.path.abspath(__file__)).parent

# Construct the absolute paths to the receipt and truck ticket directories
receipts_dir = script_dir / "receipts"
trucktickets_dir = script_dir / "trucktickets"
aws_dir = script_dir / "aws"


@dataclass
class QA:
    question: str
    valid_answers: List[str]

    def has_answer(self, response: str) -> bool:
        return any(answer in response for answer in self.valid_answers)


@dataclass
class TestCase:
    filename: Path
    qa: List[QA]

    def run(self, provider: str):
        marks = 0
        questions = [qa.question for qa in self.qa]
        print(f"Running test for {self.filename}")
        print(f"Questions: {questions}")
        answers = ocr(self.filename, questions, provider=provider)
        for i, a in enumerate(answers):
            print(f"Answer for QA {i+1}: {a}")
            if self.qa[i].has_answer(a):
                print(f"Found valid answer for QA {i+1}")
                print(f"Question: {self.qa[i].question}")
                print(f"Valid Answers: {self.qa[i].valid_answers}")
                print()
                marks += 1
            else:
                extracted_numbers = extract_and_format_numbers(a)
                print(f"Extracted numbers: {extracted_numbers}")
                for n in extracted_numbers:
                    if self.qa[i].has_answer(str(n)):
                        print(f"Found valid answer for QA {i+1}")
                        print(f"Question: {self.qa[i].question}")
                        print(f"Valid Answers: {self.qa[i].valid_answers}")
                        print(f"Extracted number: {n}")
                        print()
                        marks += 1
                        break

        return marks


RECEIPT_TESTS = [
    TestCase(
        filename=receipts_dir / "grocery1.jpg",
        qa=[QA(question="What is the bill total?", valid_answers=["203.07"])],
    ),
    TestCase(
        filename=receipts_dir / "img1.png",
        qa=[QA(question="What is the bill total?", valid_answers=["1591600"])],
    ),
    TestCase(
        filename=receipts_dir / "img3.png",
        qa=[QA(question="What is the bill total?", valid_answers=["75000"])],
    ),
    TestCase(
        filename=receipts_dir / "img4.png",
        qa=[QA(question="What is the bill total?", valid_answers=["93500"])],
    ),
    TestCase(
        filename=receipts_dir / "img5.png",
        qa=[QA(question="What is the bill total?", valid_answers=["54000"])],
    ),
    TestCase(
        filename=receipts_dir / "img6.png",
        qa=[QA(question="What is the bill total?", valid_answers=["365000"])],
    ),
    TestCase(
        filename=receipts_dir / "img7.png",
        qa=[QA(question="What is the bill total?", valid_answers=["17000"])],
    ),
    TestCase(
        filename=receipts_dir / "img8.png",
        qa=[QA(question="What is the bill total?", valid_answers=["47000"])],
    ),
    TestCase(
        filename=receipts_dir / "img10.png",
        qa=[QA(question="What is the bill total?", valid_answers=["20"])],
    ),
    TestCase(
        filename=receipts_dir / "img12.png",
        qa=[QA(question="What is the bill total?", valid_answers=["48"])],
    ),
]

TRUCKTICKET_TESTS = [
    TestCase(
        filename=trucktickets_dir / "1.webp",
        qa=[
            QA(question="What is the net payload?", valid_answers=["13180"]),
            QA(
                question="What is the net payload unit?",
                valid_answers=["kg", "KG", "kilograms", "Kilograms"],
            ),
            QA(question="What is the gross payload?", valid_answers=["32350"]),
            QA(question="What is the ticket number?", valid_answers=["22837"]),
            QA(question="What is the license plate?", valid_answers=["6637DN"]),
            QA(question="What is the truck identifier?", valid_answers=["BERTRAM14"]),
            QA(
                question="What is the material being delivered?",
                valid_answers=["BOULDERS", "GRANITE / LIMESTONE BOULDERS"],
            ),
            QA(question="How many loads were delivered so far?", valid_answers=["9"]),
        ],
    ),
]

# https://aws.amazon.com/blogs/machine-learning/specify-and-extract-information-from-documents-using-the-new-queries-feature-in-amazon-textract/
# AWS: 0.015 for queries 0.025 for custom queries

# TOTAL COST: 0.01503
AWS_VACCINATION_TESTS = [
    TestCase(
        filename=aws_dir / "vaccination.jpg",
        qa=[
            # specifying the label works but, otherwise it seems it takes first name is taken as the first name it encounters rather than the label.
            QA(question="What is the label first name value", valid_answers=["Major"]),
            QA(question="What is the label last name value", valid_answers=["Mary"]),
            QA(
                question="Which clinic site was the 1st dose COVID-19 administrated?",
                valid_answers=["XYZ"],
            ),
            QA(
                question="Who is the manufacturer for 1st dose of COVID-19?",
                valid_answers=["Pfizer"],
            ),
            QA(
                question="What is the date for the 2nd dose covid-19?",
                valid_answers=["2/8/2021", "2021-02-08", "February 8, 2021"],
            ),
            QA(question="What is the patient number", valid_answers=["012345abcd67"]),
            QA(
                question="Who is the manufacturer for 2nd dose of COVID-19?",
                valid_answers=["Pfizer"],
            ),
            QA(
                question="Which clinic site was the 2nd dose covid-19 administrated?",
                valid_answers=["CVS"],
            ),
            QA(
                question="What is the lot number for 2nd dose covid-19?",
                valid_answers=["BB5678"],
            ),
            QA(
                question="What is the date for the 1st dose covid-19?",
                valid_answers=["1/18/21", "2021-01-18", "January 18, 2021"],
            ),
            QA(
                question="What is the lot number for 1st dose covid-19?",
                valid_answers=["AA1234"],
            ),
            QA(question="What is the MI?", valid_answers=["M"]),
            QA(question="MI?", valid_answers=["M"]),
        ],
    ),
]

# TOTAL COST: 0.01119
AWS_INSURANCE_TEST = [
    TestCase(
        filename=aws_dir / "insurance.png",
        qa=[
            QA(question="What is the insured name?", valid_answers=["Jacob Michael"]),
            QA(
                question="What is the level of benefits?",
                valid_answers=["SILVER", "Silver"],
            ),
            QA(
                question="What is medical insurance provider?",
                valid_answers=["AnyInsurance Co."],
            ),
            QA(question="What is the OOP max?", valid_answers=["$6000/$12000"]),
            QA(
                question="What is the effective date?",
                valid_answers=["11/02/2021", "2021-11-02", "November 2, 2021"],
            ),
            QA(
                question="What is the office visit copay?",
                valid_answers=["$55/0%", "55"],
            ),
            QA(
                question="What is the specialist visit copay?",
                valid_answers=["$65/0%", "65"],
            ),
            QA(question="What is the member id?", valid_answers=["XZ 9147589652"]),
            QA(question="What is the plan type?", valid_answers=["AnyPlan X-EPO"]),
            QA(question="What is the coinsurance amount?", valid_answers=["30%"]),
        ],
    ),
]

# TOTAL COST: 0.0111
AWS_MORTGAGE_TEST = [
    TestCase(
        filename=aws_dir / "mortgage.jpg",
        qa=[
            QA(
                question="When is this document dated?",
                valid_answers=["March 4, 2022", "3/4/22"],
            ),
            QA(
                question="What is the note date?",
                valid_answers=["March 4, 2022", "3/4/22"],
            ),
            QA(
                question="When is the Maturity date the borrower has to pay in full?",
                valid_answers=["April, 2032", "4/2032", "April 1, 2032", "4/1/32"],
            ),
            QA(
                question="What is the note city and state?",
                valid_answers=["Anytown, ZZ"],
            ),
            QA(
                question="What is the yearly interest rate?",
                valid_answers=["4.150%", "4.150"],
            ),
            QA(question="Who is the lender?", valid_answers=["AnyCompany"]),
            QA(
                question="When does payments begin?",
                valid_answers=["April, 2022", "4/2022", "April 1, 2022", "4/1/22"],
            ),
            QA(
                question="What is the beginning date of payment?",
                valid_answers=["April, 2022", "4/2022", "April 1, 2022", "4/1/22"],
            ),
            QA(
                question="What is the initial monthly payments?", valid_answers=["2500"]
            ),
            QA(
                question="What is the interest rate?", valid_answers=["4.150%", "4.150"]
            ),
            QA(
                question="What is the principal amount borrower has to pay?",
                valid_answers=["500000"],
            ),
        ],
    ),
]

# TOTAL COST: 0.00966
AWS_PAYSTUB_TEST = [
    TestCase(
        filename=aws_dir / "paystub.jpg",
        qa=[
            QA(
                question="What is the year to date gross pay",
                valid_answers=["23526.80", "$23526.80", "23526.8", "$23526.8"],
            ),
            QA(
                question="What is the current gross pay",
                valid_answers=["452.43", "$452.43"],
            ),
        ],
    ),
]


def run_tests(
    run_receipt_tests=False,
    run_truckticket_tests=False,
    run_paystub_tests=False,
    run_mortgage_tests=False,
    run_insurance_tests=False,
    run_vaccination_tests=False,
    provider="openai",
):
    if run_receipt_tests:
        marks, total = 0, 0
        for t in RECEIPT_TESTS:
            total += len(t.qa)
            marks += t.run(provider)
        print(f"Receipt Tests: {marks}/{total} ({marks/total*100:.2f}% accuracy)")

    if run_truckticket_tests:
        marks, total = 0, 0
        for t in TRUCKTICKET_TESTS:
            total += len(t.qa)
            marks += t.run(provider)
        print(f"Truck Ticket Tests: {marks}/{total} ({marks/total*100:.2f}% accuracy)")

    if run_vaccination_tests:
        marks, total = 0, 0
        for t in AWS_VACCINATION_TESTS:
            total += len(t.qa)
            marks += t.run(provider)
        print(f"Vaccination Tests: {marks}/{total} ({marks/total*100:.2f}% accuracy)")

    if run_mortgage_tests:
        marks, total = 0, 0
        for t in AWS_MORTGAGE_TEST:
            total += len(t.qa)
            marks += t.run(provider)
        print(f"Mortgage Tests: {marks}/{total} ({marks/total*100:.2f}% accuracy)")

    if run_insurance_tests:
        marks, total = 0, 0
        for t in AWS_INSURANCE_TEST:
            total += len(t.qa)
            marks += t.run(provider)
        print(f"Insurance Tests: {marks}/{total} ({marks/total*100:.2f}% accuracy)")

    if run_paystub_tests:
        marks, total = 0, 0
        for t in AWS_PAYSTUB_TEST:
            total += len(t.qa)
            marks += t.run(provider)
        print(f"Paystub Tests: {marks}/{total} ({marks/total*100:.2f}% accuracy)")


if __name__ == "__main__":
    fire.Fire(run_tests)
