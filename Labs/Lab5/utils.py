from sklearn.tree import export_graphviz
from subprocess import Popen
from tempfile import NamedTemporaryFile

from scipy.misc import imread, imshow

def sort_features(feature_names, feature_importances):
    """Sorts features by importance.

    Arguments:
        feature_names(list): A list of strings containing the names
                             of the features.
        feature_importances(list): A list of the importances of the features.

    Returns:
        A list of (feature_name, feature_importance) tuples sorted in order
        of descending importance.
    """

    return sorted(zip(feature_names,
                      feature_importances),
                  key=lambda x: x[1],
                  reverse=True)

def display_decision_tree(tree, feature_names, class_names):
    """Displays a decision tree using graphviz.

    Arguments:
        tree(object): A trained sklearn DecisionTreeClassifier.
        feature_names(list): A list of the names of the features.
        class_names(list): A list of the names of the classes.
    """

    with NamedTemporaryFile(suffix='.dot') as dot_file:
        export_graphviz(decision_tree=tree,
                        out_file=dot_file.name,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True)
        with NamedTemporaryFile(suffix='.png') as png_file:
            Popen(['dot', '-Tpng', dot_file.name, '-o', png_file.name]).wait()
            imshow(imread(png_file.name))
