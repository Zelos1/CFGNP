
import argparse
from cfgnp.classifier.image_classifier import train_classifier
from cfgnp.classifier.train_xray_classifier_augmented import train_classifier as train_classifier_augmented
from cfgnp.util.util import NUM_CLASS_CHEXPERT, CLASS_MAPPING


def main(name):
    print(name)
    # train_classifier(name, CLASS_MAPPING.get(name), NUM_CLASS_CHEXPERT.get(name))
    train_classifier_augmented(name, CLASS_MAPPING.get(name), NUM_CLASS_CHEXPERT.get(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an X-ray classifier.")
    parser.add_argument("--name", choices=["Sex", "Age", "AP/PA", "No Finding"], type=str, help="Name of the classifier target", default="Age")
    args = parser.parse_args()

    main(args.name)