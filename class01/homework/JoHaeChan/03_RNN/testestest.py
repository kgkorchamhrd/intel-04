import FinanceDataReader as fdr

df = fdr.DataReader('005930')  # 기본적으로 전체 데이터
print(df.tail())
