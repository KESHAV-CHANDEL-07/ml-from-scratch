def predict(outlook, humidity, windy):
    """
    outlook: 0=Sunny, 1=Overcast, 2=Rain
    humidity: 0=High, 1=Normal
    windy: 0=False, 1=True
    """
    if outlook == 0:  # Sunny
        return 0  # PlayTennis = No
    elif outlook == 1:  # Overcast
        return 1  # PlayTennis = Yes
    elif outlook == 2:  # Rain
        if windy == 0:
            return 1  # PlayTennis = Yes
        else:
            return 0  # PlayTennis = No

# testing:
test_data = [
    (0, 0, 0),  # Sunny, High, False
    (1, 1, 1),  # Overcast, Normal, True
    (2, 1, 0),  # Rain, Normal, False
    (2, 0, 1)   # Rain, High, True
]

for features in test_data:
    result = predict(*features)
    if result == 1:
        result = "Yes"
    else:
        result = "No"
    print(f"features={features} => PlayTennis={result}")
