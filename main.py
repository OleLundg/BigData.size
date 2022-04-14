from Models import load_csv, KNN


def main():
    height = int(input('Enter your height in mm: '))
    weight = int(input('Enter your weight hg: '))
    gender = input('Enter your sex: ')

    X = [[height, weight]]

    X_test, y_test = load_csv(gender)
    clf = KNN(k=5)
    clf.fit(X_test, y_test)
    prediction = clf.predict(X)

    if gender == 'male':
        if prediction == 1:
            print('t-shirt size: S')
        elif prediction == 2:
            print('t-shirt size: M')
        elif prediction == 3:
            print('t-shirt size: L')
        elif prediction == 4:
            print('t-shirt size: XL')
        elif prediction == 5:
            print('t-shirt size: XXL')
    elif gender == 'female':
        if prediction == 1:
            print('t-shirt size: XS')
        elif prediction == 2:
            print('t-shirt size: S')
        elif prediction == 3:
            print('t-shirt size: M')
        elif prediction == 4:
            print('t-shirt size: L')
        elif prediction == 5:
            print('t-shirt size: XL')
        elif prediction == 6:
            print('t-shirt size: XXL')


if __name__ == '__main__':
    main()
