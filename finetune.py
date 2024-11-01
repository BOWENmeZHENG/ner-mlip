import logging
import sys
import os
from dataclasses import dataclass, field
import random
import utils as ut
import net
from trainer import train
from transformers import HfArgumentParser, BertForMaskedLM, BertTokenizer
import warnings
warnings.filterwarnings('ignore')


CLASSES = {'MATERIAL': 1, 'MLIP': 2, 'PROPERTY': 3, 'PHENOMENON': 4, 'SIMULATION': 5, 'VALUE': 6, 'APPLICATION': 7, 'O': 0}

@dataclass
class ModelArguments:
    model_name: str = field(
        default='pranav-s/MaterialsBERT',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    database_path: str = field(
        default='./instance/AnnoApp.sqlite',
        metadata={"help": "Path to database of annotations for MLIP texts (training & ID sets)."}
    )
    database_ood_1_path: str = field(
        default='./instance/AnnoApp_ood_1.sqlite',
        metadata={"help": "Path to database of annotations for nanomaterial simulation and ML texts (OOD test 1 set)."}
    )
    database_ood_2_path: str = field(
        default='./instance/AnnoApp_ood_2.sqlite',
        metadata={"help": "Path to database of annotations for optical metamaterials texts (OOD test 2 set)."}
    )

@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='./output',
        metadata={"help": "The output directory for logs."}
    )
    n_epochs: int = field(
        default=6,
        metadata={"help": "Total number of training epochs to perform."}
    )
    classes_weights: tuple = field(
        default=(0.5, 1., 1., 1., 1., 1., 0.5, 0.5),
        metadata={"help": "weights for each NER class. The first entry is the OTHER class"}
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for training."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    learning_rate: float = field(
        default=0.00005,
        metadata={"help": "learning rate for training."}
    )
    linear_probe: bool = field(
        default=False,
        metadata={"help": "Whether to using linear probing method, i.e. only update the last-layer weights."}
    )
    training_percentage: float = field(
        default=0.9,
        metadata={"help": "Percentage of total data allocated for training."}
    )
    training_actual: int = field(
        default=72,
        metadata={"help": "Number of training data that are actually used."}
    )
    seed: int = field(
        default=3242,
        metadata={"help": "Random seed for everything except data shuffling."}
    )
    seed_shuffle: int = field(
        default=5634,
        metadata={"help": "Random seed for data shuffling."}
    )

@dataclass
class OtherArguments:
    plot: bool = field(
        default=True,
        metadata={"help": "Whether to plot results."}
    )
    save_model: bool = field(
        default=True,
        metadata={"help": "Whether to save the model parameters."}
    )
    save_results: bool = field(
        default=True,
        metadata={"help": "Whether to save the prediction results."}
    )
    view_test: bool = field(
        default=True,
        metadata={"help": "Whether to view a random test data."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, OtherArguments))
    model_args, training_args, other_args = parser.parse_args_into_dataclasses()

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.FileHandler(f'{training_args.output_dir}/finetune.log')],
    )
    logging.info(f"Parameters {model_args}, {training_args}, {other_args}")

    # Shuffle data before training/test partition.
    ut.seed_everything(training_args.seed_shuffle)

    # Get training data from database
    records_id = ut.get_data(model_args.database_path)
    record_list_id = ut.form_record_list(records_id)

    # Training/test partition
    random.shuffle(record_list_id)
    N_train = int(training_args.training_percentage * len(record_list_id))
    record_list_train = record_list_id[:N_train][:training_args.training_actual]
    record_list_test = record_list_id[N_train:]
    print(f'Number of total training data: {N_train}')
    print(f'Number of training data used: {len(record_list_train)}')
    print(f'Number of test data: {len(record_list_test)}')

    # Load OOD data
    records_ood_1 = ut.get_data(model_args.database_ood_1_path)
    record_list_ood_1 = ut.form_record_list(records_ood_1)
    print(f'Number of OOD_1 data: {len(records_ood_1)}')
    records_ood_2 = ut.get_data(model_args.database_ood_2_path)
    record_list_ood_2 = ut.form_record_list(records_ood_2)
    print(f'Number of OOD_2 data: {len(records_ood_2)}')

    # Load pretrained model
    ut.seed_everything(training_args.seed)
    tokenizerBERT = BertTokenizer.from_pretrained(model_args.model_name, model_max_length=training_args.max_seq_length)
    modelBERT = BertForMaskedLM.from_pretrained(model_args.model_name)
    model = net.NERBERTModel(modelBERT.base_model, output_size=len(CLASSES)+1)

    # Run training
    pred_test, pred_ood_1, pred_ood_2 = train(
        model, tokenizerBERT,
        record_list_train, record_list_test, record_list_ood_1, record_list_ood_2, CLASSES, 
        training_args.train_batch_size, training_args.seed, training_args.max_seq_length, training_args.classes_weights, 
        training_args.learning_rate, training_args.n_epochs, training_args.linear_probe,
        plot=other_args.plot, save_model=other_args.save_model, save_results=other_args.save_results
        )

    # Quick view of test annotation
    if other_args.view_test:
        sample_test_id = random.randint(0, len(record_list_test) - 1)
        word_test_list = record_list_test[sample_test_id]['words']
        predictions_test = pred_test[sample_test_id, :, :].max(dim=0)[1]
        real_preds_test = predictions_test[:len(word_test_list)]
        ut.show_pred(real_preds_test, word_test_list)

        sample_ood_1_id = random.randint(0, len(record_list_ood_1) - 1)
        word_ood_1_list = record_list_ood_1[sample_ood_1_id]['words']
        predictions_ood_1 = pred_ood_1[sample_ood_1_id, :, :].max(dim=0)[1]
        real_preds_ood_1 = predictions_ood_1[:len(word_ood_1_list)]
        ut.show_pred(real_preds_ood_1, word_ood_1_list)

        sample_ood_2_id = random.randint(0, len(record_list_ood_2) - 1)
        word_ood_2_list = record_list_ood_2[sample_ood_2_id]['words']
        predictions_ood_2 = pred_ood_2[sample_ood_2_id, :, :].max(dim=0)[1]
        real_preds_ood_2 = predictions_ood_2[:len(word_ood_2_list)]
        ut.show_pred(real_preds_ood_2, word_ood_2_list)

if __name__ == "__main__":
    main()