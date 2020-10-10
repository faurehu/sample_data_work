import pandas as pd

columns = {
    '[run number]': 'run',
    'i-n-parties': 'n_parties',
    'i-party-strategy': 'strategy',
    '[step]': 'step',
    'i-global-q': 'q',
    'i-lambda': 'lambd',
    'i-kappa': 'kappa',
    'o-total-votes              ; count of votes': 'votes',
    'o-mean-current-u           ; mean voter utility': 'mean_loss',
    'o-mean-current-precise-u   ; mean voter utility when voters voter correctly': 'mean_precise_loss',
    'o-voter-misery             ; mean quadratic Euclidean voter distance from closest party': 'voter_misery',
    'o-enp                      ; effective number of parties = 1/(sum of squared party vote shares)': 'enp',
    'o-incorrect-cost           ; difference between total utility of correct voting and incorrect voting': 'incorrect_cost',
    'o-p-eccentricity           ; mean Euclidean distance of parties from (mean-voter-x) on x axis / 10': 'p_eccentricity',
    'o-party-q-eccentricity     ; mean Euclidean distance of parties from (mean-voter-y) on y axis / 10': 'q_eccentricity',
    'o-mean-eccentricity        ; mean Euclidean distance of parties from (mean-voter-x, mean-voter-y) / 10': 'eccentricity',
    'o-global-q-eccentricity    ; mean distance from global-q': 'global_q_eccentricity',
    'o-p-dist                   ; x distance between parties (only when n-parties = 2)': 'p_dist',
    'o-q-dist                   ; y distance between parties (only when n-parties = 2)': 'q_dist',
    'o-win-q-dist               ; y distance between winner and global-q': 'win_q_dist',
    'o-pct-incorrect            ; percentage of voters voting incorrectly': 'pct_incorrect',
    'o-vote-share-diff          ; difference in vote share percentage': 'share_diff'
}


def get_data():
    df = pd.read_csv(
        './data/Electoral Competition Model all-strat symmetric-table.csv', header=6)
    return df[list(columns.keys())].rename(columns=columns)
