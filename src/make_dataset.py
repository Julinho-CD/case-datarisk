from src.config import TEST_PATH, TRAIN_PATH
from src.data_access import (
    build_train_test,
    clean_types,
    create_target,
    deduplicate,
    load_raw_case_data,
    save_processed_datasets,
    validate_case_alignment,
)


def main():
    base_cadastral, base_info, base_pag_dev, base_pag_test = load_raw_case_data()
    base_cadastral, base_info, base_pag_dev, base_pag_test = clean_types(
        base_cadastral, base_info, base_pag_dev, base_pag_test
    )
    base_pag_dev, base_pag_test = deduplicate(base_pag_dev, base_pag_test)
    base_pag_dev = create_target(base_pag_dev)

    train, test = build_train_test(base_cadastral, base_info, base_pag_dev, base_pag_test)
    validate_case_alignment(base_pag_dev, base_pag_test, train, test)
    save_processed_datasets(train, test)

    print(f"OK: saved {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()
