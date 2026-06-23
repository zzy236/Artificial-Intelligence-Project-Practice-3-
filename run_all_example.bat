@echo off
REM Windows 示例运行脚本。请先把 RAVDESS wav 文件放到 data\raw\RAVDESS\ 下。

python src\01_parse_metadata.py --raw_dir data\raw\RAVDESS --out_csv data\processed\ravdess_metadata.csv

python src\02_extract_features.py --metadata_csv data\processed\ravdess_metadata.csv --out_csv data\processed\ravdess_acoustic_features.csv

python src\03_statistical_analysis.py --features_csv data\processed\ravdess_acoustic_features.csv --out_dir results\tables

python src\04_train_baseline_models.py --features_csv data\processed\ravdess_acoustic_features.csv --out_dir results\tables --task four_class --cv groupkfold --n_splits 6

python src\05_visualize_results.py --features_csv data\processed\ravdess_acoustic_features.csv --anova_csv results\tables\anova_results.csv --classification_csv results\tables\classification_results.csv --predictions_csv results\tables\overall_predictions.csv --fig_dir results\figures

echo Done.
pause
