from src.fact_verification_dataset import MarriageFactVerificationDataset
from src.tacred_spouse_fact_verification_dataset import TacredSpouseFactVerificationDataset
from src.tacred_spouse_fact_verification_dataset_combined import TacredSpouseFactVerificationCombinedDataset


def get_unified_qa_dataset(dataset_name):
    train_file = f'data/unifiedQA/{dataset_name}.json'
    return MarriageFactVerificationDataset(train_file)


def get_tacred_dataset(dataset_name):
    return TacredSpouseFactVerificationDataset('./data/tacred', f'{dataset_name}.txt')


def get_tacred_combined_dataset(dataset_name):
    return TacredSpouseFactVerificationCombinedDataset('./data/tacred', f'{dataset_name}.txt')


def get_anchors_dataset(dummy):
    return MarriageFactVerificationDataset()


get_dataset_functions = {
    "unifiedQA": get_unified_qa_dataset,
    "tacred": get_tacred_dataset,
    "anchor": get_anchors_dataset,
    "tacred_combined": get_tacred_combined_dataset
}
