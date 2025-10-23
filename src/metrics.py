from sklearn.metrics import accuracy_score, precision_score, recall_score

def get_metrics(y, y_pred):
    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred) * 100
    recall = recall_score(y, y_pred) * 100

    print(f"""
        accuracy score: {accuracy:.2f}%
        precision score: {precision:.2f}%"
        recall score: {recall:.2f}%
    """)

    return [accuracy, precision, recall]

