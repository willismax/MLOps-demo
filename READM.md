# MLOps Example with CML

## 說明
本範例示範由Github Actions 整合深度學習的 CI/CD 流程。
相關技術:
- 深度學習框架: [TensorFlow](https://www.tensorflow.org/)
- 資料集: [fashion_mnist](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- 混淆矩陣: [scikit-learn](https://scikit-learn.org/stable/)
- 繪圖: [Matplotlib](https://matplotlib.org/)
- 開發環境: [pipenv](https://pipenv.pypa.io/en/latest/)
- MLOps: 
  - Continuous Machine Learning: [CML](https://cml.dev/)
  - CI/CD: [Github Actions](https://github.com/features/actions)

## 如何使用

- Fork到自己的Repo。
- 本機 `git clone https://github.com/{您的Github帳號}/mlops-demo.git` ，本機須安裝 [git](https://git-scm.com/)。
    ```
    git clone https://github.com/{YOUR_GITHUB_NAME}/mlops-demo.git
    cd mlops-demo
    ```
- 本機環境安裝，在此示範用 [pipenv](https://pipenv.pypa.io/en/latest/) 建立環境。
    ```
    pipenv --python 3.x  #建立python 3.x版虛擬環境
    pipenv sync  #從Pipfile.lock同步安裝模組
    pipenv shell  #進入虛擬環境
    python model.py  #執行主程式
    
    exit  # 離開虛擬環境
    pipenv --rm  # 移除虛擬環境
    ```

- 本機修正部分程式碼後(譬如調整epochs次數)，執行 `git push`。
    ```
    git add .
    git commit -am "Fix: model.py"
    git push  #預設至origin main
    ```
- 觀察 Github Actions。至 `https://github.com/{YOUR_GITHUB_NAME}/mlops-demo/actions` 查看執行過程。
  ![](https://i.imgur.com/ctDyfg6.png)

- 您也會收到由 Github Actions 寄的 email 報告。

## 程式重點

### 使用 Github Actions
- 在 `.github/workflows/cml.yaml` 以 YAML 設定 Github Actions 環境與執行工作。
- 更多 GitHub Actions 操作參見 [GitHub Actions 官方文件](https://docs.github.com/en/actions/quickstart)。
    ```
    name: mlops-tensorflow-mnist-example
    on: [push]
    jobs:
    run:
        runs-on: [ubuntu-latest]
        container: docker://dvcorg/cml-py3:latest
        steps:
        - uses: actions/checkout@v2
        - name: 'Train model'
            env:
            REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            run: |
            # Your ML workflow goes here
            pip install -r requirements.txt
            python model.py
        - name: Write CML report
            env:
            REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            run: |
            echo "## Post reports as comments in GitHub PRs" > report.md
            cat result.txt >> report.md
            echo "\n## Model Performance" >> report.md
            echo "Model performance metrics are on the plot below." >> report.md
            cml-publish plot_confusion_matrix.png --md >> report.md
            cml-publish plot_loss.png --md >> report.md
            cml-publish plot_accuracy.png --md >> report.md
            cml-send-comment report.md
    ```

## Keywords
MLOps, Machine Learning, Data Science