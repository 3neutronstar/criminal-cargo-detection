import numpy as np


# Implementation
#    If the number of times that the companies or people commit crimes is smaller than threshlod, those classification codes would be masked to 'Missing'.

# parameters
#    in_column : ndarray type column data
#    threshold : int type variable, which decide to mask the classification code

# return
#   in_column_clone : ndarray type column data, which is masked

def get_list(in_column, threshold):

    crime_idx = np.where(in_column == 1)[0]
    non_crime_idx = np.where(in_column == 0)[0]

    # 우범여부 masking
    only_crime = in_column[crime_idx].reshape(20560,)

    # 높은 우범횟수 순으로 정렬
    elem, count = np.unique(only_crime, return_counts=True)

    column_dict = dict(zip(elem, count))

    column_sorted = sorted(column_dict.items(), reverse=True, key = lambda x : x[1])
    column_sorted_np = np.array(column_sorted)

    # numpy로 형변환
    # code : str type classification codes
    # counts : the number of times each code has been counted
    code, counts = column_sorted_np[:, 0], column_sorted_np[:, 1]

    # counts 형변환
    counts = counts.astype(np.uint64)
    where_clip = np.argmin(counts > threshold)

    bigger_than_threshold = code[:where_clip]

    in_column_clone = in_column.copy()

    for idx, r in enumerate(in_column) :

        # Nan 처리 해줬으면 필요없음
        # if type(r) is float :
        #     in_column_clone[idx] = 'Missing'
        #     print('nan')

        if r not in bigger_than_threshold :
            in_column_clone[idx] = 'Missing'

        if idx in non_crime_idx[0]:
            in_column_clone[idx] = 'Missing'

    return in_column_clone 