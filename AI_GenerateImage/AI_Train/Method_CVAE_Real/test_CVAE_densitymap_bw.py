from cvae_density_test import main


if __name__ == "__main__":
    main(
        default_config="config_test_densitymap_bw.json",
        target_representation="bw",
        target_channels=3,
    )
