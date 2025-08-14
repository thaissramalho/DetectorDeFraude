from src.model_training import train_model
from src.model_inference import predict_new_transactions

def main():
    print("🔧 Treinando o modelo...")
    train_model()

    print("📈 Fazendo previsões em novos dados...")
    result = predict_new_transactions()
    print(result.head())

if __name__ == '__main__':
    main()