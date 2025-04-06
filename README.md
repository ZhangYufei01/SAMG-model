# SAMG-Net: Leveraging Semi-Automatic Mask Guidance for Enhanced 3D Tumor Segmentation in Medical Imaging

This project contains code for performing a specific operation. Below are the steps for calling the code:

## Installation

To install the required libraries for this project, please refer to the `requirements.txt` file.

## Steps

1. Set the Python path:

    ```bash
    export PYTHONPATH="/dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG:$PYTHONPATH"
    ```

2. Preprocessing:

    ```bash
    python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m preprocess --pre /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data
    ```

3. Train the model:

    ```bash
    python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m train --d /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/fold_0.json --o /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0 --pd /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/preprocess_data --f 0 --p "(64,192,192)"
    ```

4. Model prediction:

    ```bash
    python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m predict --d /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/predict_semi.json --o /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict --f 0 --c /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/checkpoint_best.pth --p "(64,192,192)"
    ```

5. Model evaluation:

    ```bash
    python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m evaluation --g /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/labelsTs --op /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/0 --j /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/0/comparison_results.json
    ```

6. Post-processing:

    ```bash
    python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m postprocess --ip /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/0 --op /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/post_0 --x /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/cancer_infor_cutpage_big_tumour_test.xlsx
    ```

## Dependencies

The dependencies for this project are listed in the `requirements.txt` file. Make sure to install these dependencies.
