from cvae_density_train import main


if __name__ == "__main__":
    main(
        default_config="config_train_real_cvae.json",
        target_representation="bw",
        target_channels=1,
        test_script_name="test_CVAE_real_densitymap_bw.py",
    )
