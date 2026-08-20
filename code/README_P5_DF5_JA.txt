V12 p=5, nu=5: standardized-shape proposal vs DirectLiNGAM and ICA-LiNGAM

実験設定
--------
- p=5
- n=50, 100, 200
- nu=5
- 5条件（交絡なし、隣接/非隣接 gamma=0.4, 0.8）
- 100反復（既定値）
- 比較方法:
  * proposed_standardized_shape
  * direct_lingam
  * ica_lingam

Google Colabでの実行
--------------------
1) full_optimizer_bayesian_ica_v12_p5_standardized_direct_ica_df5.zip をアップロードする。
2) 次を実行する。

   !rm -rf /content/full_optimizer_bayesian_ica_v12_p5_standardized_direct_ica_df5
   !unzip -q -o /content/full_optimizer_bayesian_ica_v12_p5_standardized_direct_ica_df5.zip -d /content/

3) 次を実行する。

   %run /content/full_optimizer_bayesian_ica_v12_p5_standardized_direct_ica_df5/colab_p5_df5_standardized_direct_ica_runner.py

4) 入力例

   target total number of replications [100]: 100
   master seed [20260718]: 20260718
   number of worker processes [2]: 2

再開
----
- 中断後は、同じrunnerを同じtarget、seedで再実行する。
- 出力prefixは固定されているため、完了済みの行は自動的にスキップされる。
- --overwrite は付けない。
- 100回完了後に200回へ追加する場合はtarget=200、同じseedを入力する。

出力先
------
/content/drive/MyDrive/BayesianICA/results/

主な出力
--------
- p5_seed20260718_df5_standardized_direct_ica_v12_raw.csv
- p5_seed20260718_df5_standardized_direct_ica_v12_summary.csv
- p5_seed20260718_df5_standardized_direct_ica_v12_overall_summary.csv
- p5_seed20260718_df5_standardized_direct_ica_v12_paired_comparisons.csv
- p5_seed20260718_df5_standardized_direct_ica_v12_config.json
