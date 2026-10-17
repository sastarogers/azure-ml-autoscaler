import math


def desired_replicas(
    predicted_cpu,
    min_replicas=1,
    max_replicas=3,
    target_cpu=60,
):
    count = math.ceil(max(0, predicted_cpu) / target_cpu)
    return max(min_replicas, min(max_replicas, count))


if __name__ == "__main__":
    for prediction in [20, 45, 70, 95]:
        print(
            f"Predicted CPU: {prediction}% -> "
            f"recommended replicas: "
            f"{desired_replicas(prediction)}"
        )
