from cvae_density_test import main


if __name__ == "__main__":
    main(
        default_config="config_test_real_cvae.json",
        target_representation="bw",
        target_channels=1,
    )
