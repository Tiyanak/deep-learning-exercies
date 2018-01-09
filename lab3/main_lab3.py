from lab3.data_processor import Dataset
from lab3.Language_model import LanguageModel

DATA_DIR = "dataset/selected_conversations.txt"

def main():

    dataset = Dataset()

    dataset.preprocess(DATA_DIR)

    langModel = LanguageModel()

    langModel.run_language_model(dataset, 30)

if __name__ == "__main__":

    main()

