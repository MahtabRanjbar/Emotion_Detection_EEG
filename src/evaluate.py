def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = (total_correct / total_samples) * 100
    return avg_loss, accuracy, predictions, targets


# Inference function to evaluate the test dataset and print classification metrics.
def test_model(model, model_type, output_dir, test_loader, device, criterion):
    test_loss, test_accuracy, pred, target = evaluate(
        model, test_loader, device, criterion
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Further metrics, such as precision, recall, f1-score can be obtained by making predictions on all data
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    result = dict()
    precision = precision_score(all_targets, all_predictions, average="weighted")
    recall = recall_score(all_targets, all_predictions, average="weighted")
    f1 = f1_score(all_targets, all_predictions, average="weighted")

    # Store in history
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["test_Loss"] = test_loss
    result["test_accuracy"] = test_accuracy

    print(
        f"Test Precision: {precision:.2f}, Test Recall: {recall:.2f}, Test F1 Score: {f1:.2f}"
    )

    log_path = f"{output_dir}/{model_type}_evaluation_results.txt"
    with open(log_path, "w") as log_file:
        log_file.write(
            f"Test accuracy: {test_accuracy:.2}Test Precision: {precision:.2f},\n Test Recall: {recall:.2f},\n Test F1 Score: {f1:.2}"
        )

    return result, pred, target
