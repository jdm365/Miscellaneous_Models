import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Node:
    def __init__(
            self,
            df,
            gradient,
            hessian,
            idxs,
            subsample_cols=0.8,
            min_data_in_leaf=5,
            min_child_weight=1.0,
            depth=10,
            l1_reg=1.0,
            l2_reg=1.0,
            eps=0.1,
            ):
        self.df               = df
        self.gradient         = gradient
        self.hessian          = hessian
        self.idxs             = idxs
        self.min_child_weight = min_child_weight
        self.min_data_in_leaf = min_data_in_leaf
        self.subsample_cols   = subsample_cols
        self.depth            = depth
        self.l1_reg           = l1_reg
        self.l2_reg           = l2_reg
        self.eps              = eps
        self.n_rows           = df.shape[0]
        self.n_cols           = df.shape[1]


        self.column_subsample_idxs = np.random.randint(
                low=0, 
                high=self.n_cols, 
                size=int(self.n_cols*subsample_cols),
                )
        self.gamma = self.compute_gamma(
                gradient=self.gradient[self.idxs], 
                hessian=self.hessian[self.idxs],
                )

        self.score = float('-inf')
        self.find_optimal_split()

    def compute_gamma(self, gradient, hessian):
        gamma = -np.sum(gradient) / (np.sum(hessian) + self.l2_reg)
        return gamma
    
    def find_optimal_split(self):
        for col_idx in self.column_subsample_idxs:
            self.find_greedy_split(col_idx)

        if self.is_leaf:
            return

        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]

        self.lhs = Node(
                df=self.df,
                gradient=self.gradient,
                hessian=self.hessian,
                idxs=self.idxs[lhs],
                min_data_in_leaf=self.min_data_in_leaf,
                depth=self.depth-1,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                min_child_weight=self.min_child_weight,
                eps=self.eps,
                subsample_cols=self.subsample_cols,
                )
        self.rhs = Node(
                df=self.df,
                gradient=self.gradient,
                hessian=self.hessian,
                idxs=self.idxs[rhs],
                min_data_in_leaf=self.min_data_in_leaf,
                depth=self.depth-1,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                min_child_weight=self.min_child_weight,
                eps=self.eps,
                subsample_cols=self.subsample_cols,
                )

    def find_greedy_split(self, col_idx):
        df = self.df[self.idxs, col_idx]

        for row in range(len(df)):
            lhs = df <= df[row]
            rhs = df > df[row]

            lhs_idxs = np.nonzero(df <= df[row])[0]
            rhs_idxs = np.nonzero(df > df[row])[0]

            cond_0 = rhs.sum() < self.min_data_in_leaf
            cond_1 = lhs.sum() < self.min_data_in_leaf
            cond_2 = self.hessian[lhs_idxs].sum() < self.min_child_weight
            cond_3 = self.hessian[rhs_idxs].sum() < self.min_child_weight
            if cond_0 or cond_1 or cond_2 or cond_3:
                continue

            current_score = self.gain(lhs, rhs)
            if current_score > self.score:
                self.col_idx = col_idx
                self.score = current_score
                self.split = df[row]

        
    def weighted_quantile_sketch(self, col_idx):
        df = self.df[self.idxs, col_idx]
        hessian_ = self.hessian[self.idxs]
        new_df = pd.DataFrame({'feature': df, 'hessian': hessian_})

        new_df.sort_values(by=['feature'], ascending=True, inplace=True)
        hessian_sum = new_df['hessian'].sum()
        f = lambda x: (1 / hessian_sum) * sum(new_df[new_df['feature'] < x['feature']['hessian']])
        new_df['rank'] = new_df.apply(f, axis=1)

        for row in range(len(new_df)):
            diff = abs(new_df['rank'].iloc[row] - new_df['rank'].iloc[row+1])

            if diff >= self.eps:
                continue

            split_values = (new_df['rank'].iloc[row] + new_df['rank'].iloc[row+1]) / 2
            lhs = df <= split_value
            rhs = df > split_value

            lhs_idxs = np.nonzero(df <= split_value)[0]
            rhs_idxs = np.nonzero(df > split_value)[0]
            
            cond_0 = rhs.sum() < self.min_data_in_leaf
            cond_1 = lhs.sum() < self.min_data_in_leaf
            cond_2 = self.hessian[lhs_idxs].sum() < self.min_child_weight
            cond_3 = self.hessian[rhs_idxs].sum() < self.min_child_weight
            if cond_0 or cond_1 or cond_2 or cond_3:
                continue

            current_score = self.gain(lhs, rhs)
            if current_score > self.score:
                self.col_idx = col_idx
                self.score = current_score
                self.split = df[row]
            

    def gain(self, lhs, rhs):
        gradient = self.gradient[self.idxs]
        hessian  = self.hessian[self.idxs]

        lhs_gradient = gradient[lhs].sum()
        lhs_hessian  = hessian[lhs].sum()

        rhs_gradient = gradient[rhs].sum()
        rhs_hessian  = hessian[rhs].sum()

        expr_0 = lhs_gradient ** 2 / (lhs_hessian + self.l2_reg)
        expr_1 = rhs_gradient ** 2 / (rhs_hessian + self.l2_reg)
        expr_2 = (lhs_gradient + rhs_gradient) ** 2 / (lhs_hessian + rhs_hessian + self.l2_reg)
        
        gain = 0.5 * (expr_0 + expr_1 - expr_2) - self.gamma
        return gain

    @property
    def split_col(self):
        return self.df[self.idxs, self.col_idx]

    @property
    def is_leaf(self):
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(val) for val in x])

    def predict_row(self, x):
        if self.is_leaf:
            return self.gamma

        node = self.lhs if x[self.col_idx] <= self.split else self.rhs
        return node.predict_row(x)


class XGBoostTree:
    def fit(
            self,
            df,
            gradient,
            hessian,
            subsample_cols=0.8,
            min_data_in_leaf=5,
            min_child_weight=1.0,
            depth=10,
            l1_reg=1.0,
            l2_reg=1.0,
            eps=0.1,
            ):
        self.tree = Node(
                df=df,
                gradient=gradient,
                hessian=hessian,
                idxs=np.arange(len(df)),
                min_data_in_leaf=min_data_in_leaf,
                depth=depth,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                min_child_weight=min_child_weight,
                eps=eps,
                subsample_cols=subsample_cols,
                )
        return self

    def predict(self, X):
        return self.tree.predict(X)


class XGBoost:
    def __init__(self, boosting_type='classifier'):
        self.estimators = []
        self.boosting_type = boosting_type
        self.vectorized_sigmoid = np.vectorize(self.sigmoid_non_vectorized)

    def sigmoid_non_vectorized(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid(self, x):
        return self.vectorized_sigmoid(x)

    def calculate_gradient(self, preds, labels):
        if self.boosting_type == 'classifier':
            preds = self.sigmoid(preds)
            return preds - labels
        if self.boosting_type == 'regression':
            return 2 * (preds - labels)

    def calculate_hessian(self, preds, labels):
        if self.boosting_type == 'classifier':
            preds = self.sigmoid(preds)
            return preds * (1 - preds)
        if self.boosting_type == 'regression':
            return np.full((preds.shape[0], 1), 2).flatten().astype('float32')

    @staticmethod
    def log_odds(col):
        true_vals   = np.count_nonzero(col == 1)
        false_vals  = np.count_nonzero(col == 0)
        return np.log(true_vals/fals_vals)

    def fit(
            self,
            df,
            y,
            lr=0.1,
            num_boosting_rounds=100,
            subsample_cols=0.8,
            min_data_in_leaf=5,
            min_child_weight=1.0,
            depth=10,
            l1_reg=1.0,
            l2_reg=1.0,
            eps=0.1,
            ):
        if type(df) == pd.DataFrame:
            df = df.to_numpy()

        self.X                   = df
        self.y                   = y
        self.min_child_weight    = min_child_weight
        self.min_data_in_leaf    = min_data_in_leaf
        self.depth               = depth
        self.l1_reg              = l1_reg
        self.l2_reg              = l2_reg
        self.eps                 = eps
        self.lr                  = lr
        self.num_boosting_rounds = num_boosting_rounds

        if self.boosting_type == 'classifier':
            self.preds = np.full((self.X.shape[0], 1), 1).flatten().astype('float32')
        if self.boosting_type == 'regression':
            self.preds = np.full((self.X.shape[0], 1), np.mean(self.y))
            self.preds = self.preds.flatten().astype('float32')

        progress_bar = tqdm(total=self.num_boosting_rounds, desc='Training')
        for booster in range(self.num_boosting_rounds):
            gradient = self.calculate_gradient(self.preds, self.y)
            hessian  = self.calculate_hessian(self.preds, self.y)
            tree     = XGBoostTree().fit(
                    df=self.X,
                    gradient=gradient,
                    hessian=hessian,
                    subsample_cols=subsample_cols,
                    min_data_in_leaf=min_data_in_leaf,
                    min_child_weight=min_child_weight,
                    depth=depth,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    eps=eps,
                    )
            self.preds += self.lr * tree.predict(self.X)
            self.estimators.append(tree)
            progress_bar.update(1)

    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])

        for estimator in self.estimators:
            pred += self.lr * estimator.predict(X)

        predicted_proba = np.full((X.shape[0], 1), 1).flatten().astype('float32')
        predicted_proba = self.sigmoid(predicted_proba + pred)
        return predicted_proba

    def predict(self, X):
        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        pred = np.zeros(X.shape[0])

        for estimator in self.estimators:
            pred += self.lr * estimator.predict(X)

        if self.boosting_type == 'classifer':
            predicted_proba = np.full((X.shape[0], 1), 1).flatten().astype('float32')
            predicted_proba = self.sigmoid(predicted_proba + pred)
            preds = np.where(predicted_proba > np.mean(predicted_proba), 1, 0)
        if self.boosting_type == 'regression':
            preds = np.full((X.shape[0], 1), np.mean(self.y))
            preds = preds.flatten().astype('float32') + pred
        return preds



if __name__ == '__main__':
    df = pd.read_csv('data/iris.csv')
    model = XGBoost(boosting_type='regression')
    features = [x for x in df.columns if x != 'target']
    #df = df[df['target'].isin([0, 1])].reset_index(drop=True)

    train_df, test_df = train_test_split(df)
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    model.fit(train_df[features], train_df['target'], num_boosting_rounds=5)
    preds = model.predict(test_df[features])
    mse = (sum(preds - test_df['target'].values) / len(test_df)) ** 2
    print(mse)

