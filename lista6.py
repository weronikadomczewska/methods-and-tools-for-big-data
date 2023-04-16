import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap


def apply_knearest_neighbours_to_dataset(dataset):
    X = dataset.data[:, :2]
    Y = dataset.target

    X_CV, X_test, Y_CV, Y_test = train_test_split(X, Y, test_size=0.1)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_CV, Y_CV)

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)

    cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
    cmap_bold = ["darkorange", "c", "darkblue"]

    fig, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        knn,
        X_test,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=dataset.feature_names[0],
        ylabel=dataset.feature_names[1],
        shading="auto",
    )

    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=dataset.target_names[Y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )

    plt.show()



iris = datasets.load_iris()
apply_knearest_neighbours_to_dataset(iris)
# wine = datasets.load_wine()
# diabetes = datasets.load_diabetes()

