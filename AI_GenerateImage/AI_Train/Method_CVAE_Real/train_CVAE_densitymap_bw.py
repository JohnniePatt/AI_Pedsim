from cvae_density_train import main


if __name__ == "__main__":
    main(
        default_config="config_train_densitymap_bw.json",
        target_representation="bw",
        target_channels=3,
        test_script_name="test_CVAE_densitymap_bw.py",
    )
