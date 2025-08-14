# Projeto: Detector de Fraudes com Machine Learning

## Descrição
Este projeto utiliza aprendizado de máquina para detectar transações financeiras suspeitas em bancos, fintechs e e-commerce. Usa XGBoost, PyCaret e técnicas de balanceamento como SMOTE.

## Tecnologias
- Python 3.10+
- scikit-learn
- xgboost
- pycaret
- imblearn (SMOTE)
- pandas, numpy, matplotlib, seaborn

## Instruções
1. Coloque os dados em `data/transactions.csv`
2. Execute o treinamento: `python src/model_training.py`
3. Faça predições: `python src/model_inference.py`

## Melhorias Futuras
- Interface com Streamlit
- Dashboard de monitoramento
- Deploy na nuvem
