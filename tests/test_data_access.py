from src.data_access import (
    DATA_SOURCE_AUTO,
    OFFICIAL_REPOSITORY_URL,
    official_file_urls,
    resolve_data_source,
)


def test_resolve_data_source_accepts_auto():
    assert resolve_data_source(DATA_SOURCE_AUTO) == DATA_SOURCE_AUTO


def test_official_file_urls_point_to_datarisk_repository():
    urls = official_file_urls("base_cadastral.csv")
    assert urls
    assert all("datarisk-case-ds-junior" in url for url in urls)
    assert OFFICIAL_REPOSITORY_URL.endswith("datarisk-case-ds-junior")
