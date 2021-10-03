import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer
from sklearn import preprocessing




import os
path=os.getcwd()
temp='//temp.csv'
path=path+temp


st.sidebar.title(
"DATA PREPROCESSOR APP" 
)	

file=st.sidebar.checkbox("File Import")

if file:

	#st.sidebar.checkbox("Choose The Format of the File",value = True)

	try:

		forma=["Xlsx","Csv"]
		upload=st.sidebar.selectbox("Choose The Format of the File ",forma)



		if upload=="Xlsx":
			uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")
			if uploaded_file is not None:
				df=pd.read_excel(uploaded_file)
				df=df.drop(df.columns[0], axis=1)
				if st.sidebar.button('Upload File'):
					st.dataframe(df)
					df.to_csv(path)

		elif upload=="Csv":
			uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")
			if uploaded_file is not None:
				df=pd.read_csv(uploaded_file)
				if st.sidebar.button('Upload File'):
				    #df=upload_xlsx(uploaded_file)
					st.dataframe(df)
					df.to_csv(path,index=False)

	except:
		pass


mt=st.sidebar.checkbox("Missing Value Treatment")

if mt:

	try:

		mvt=["Mean Imputation","Median Imputation ","Mode Imputation"]
		treatment=st.sidebar.selectbox("Choose an Option to Perform ", mvt)

		if treatment=='Mean Imputation':
			if st.sidebar.button('Process Mean'):
				#st.write(path)
				df=pd.read_csv("temp.csv")
				df=(df.fillna(df.mean()))
				df.fillna(df.select_dtypes(include='object').mode())
				df=df.drop(df.columns[0], axis=1)
				st.dataframe(df)
				df.to_csv(path)

		elif treatment=='Median Imputation':
			if st.sidebar.button('Process Median'):
				df=pd.read_csv("temp.csv")
				df=(df.fillna(df.median()))
				df=df.fillna(df.select_dtypes(include='object').mode())
				#df=df.drop(df.columns[0], axis=1)
				st.dataframe(df)
				df.to_csv(path,index=False, sep=',')

		elif treatment=='Mode Imputation':
			if st.sidebar.button('Process Mode'):
				df=pd.read_csv("temp.csv")
				df=(df.fillna(df.mode()))
				df=df.fillna(df.select_dtypes(include='object').mode())
				st.dataframe(df)
				df.to_csv(path)
	except:
		pass

FE=st.sidebar.checkbox("Feature Encoding")

if FE:

	try:

		encode=["One Hot Encoding","Dummy Encoding","Label Encoding","Hash Encoding","Frequency Encoding"]
		encoding=st.sidebar.selectbox("Choose an Option to Perform ", encode)

		if encoding=="One Hot Encoding":
			if st.sidebar.button('Process One Hot Encoding'):
				df=pd.read_csv("temp.csv")
				df=pd.get_dummies(df)
				df=df.drop(df.columns[0], axis=1)
				st.dataframe(df)
				df.to_csv(path)
				

		elif encoding=="Dummy Encoding":
			if st.sidebar.button('Process Dummy Encoding'):
				df=pd.read_csv("temp.csv")
				df=pd.get_dummies(df,drop_first=True)
				df=df.drop(df.columns[0], axis=1)
				st.dataframe(df)
				df.to_csv(path)

		elif encoding=="Label Encoding":
			if st.sidebar.button('Process Label Encoding'):
				df=pd.read_csv("temp.csv")
				df2=df.select_dtypes(include='object')
				for i in df2.columns:
					df2[i] = df2[i].astype('category')
					df2[i] = df2[i].cat.codes
				df1=df.drop(df.select_dtypes(include="object"),axis=1)
				df=pd.concat([df2, df1], axis=1)
				df=df.drop(df.columns[0], axis=1)
				st.dataframe(df)
				df.to_csv(path)
				
		elif encoding=="Frequency Encoding":
			if st.sidebar.button('Process Frequency Encoding'):
				df=pd.read_csv("temp.csv")
				df['target']=df[input("enter: ")]
				df['target']=pd.DataFrame(df['target'])
				df=pd.concat([df,df['target']],axis=1)
				te_df=df.copy()

				for col in te_df.select_dtypes(include='O').columns:
				    te=TargetEncoder()
				    te_df[col]=te.fit_transform(te_df[col],te_df.target)
				st.dataframe(te_df)
				df.to_csv(path)
	except:
		pass

OT=st.sidebar.checkbox("Outlier Treatment")

if OT:

	try:
		out=["Inter Quantile Range Method","Extreme Value Analysis"]
		outlier=st.sidebar.selectbox("Choose an Option to Perform ", out)

		if outlier=='Inter Quantile Range Method':
			if st.sidebar.button('Process IQR'):
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				df = df.select_dtypes(include=np.number)
				Q1 = df.quantile(0.25)
				Q3 = df.quantile(0.75)
				IQR = Q3 - Q1
				df= df[~((df < (Q1 - 1.5*IQR))| (df > (Q3 + 1.5*IQR))).any(axis=1)]
				st.dataframe(df)
				df.to_csv(path,index=False)

		elif outlier=='Extreme Value Analysis':
			if st.sidebar.button('Process Z Score Method'):
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				df = df.select_dtypes(include=np.number)
				std=np.std(df)
				mean=np.mean(df)
				df=df[((df-mean)/std).any(axis=1)]
				st.dataframe(df)
				df.to_csv(path,index=False)
	except:
		pass

FS=st.sidebar.checkbox("Feature Scaling")

if FS:

	try:

		std=["Standard Scaler","MinMax Scaler","Robust Scaler","MaxAbs Scaler"]
		scaling=st.sidebar.selectbox("Choose an Option to Perform ", std)

		if scaling=='Standard Scaler':
			if st.sidebar.button('Process Standard Scaler'):
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				X = df.select_dtypes(include=np.number)
				mean_X = np.mean(X)
				std_X = np.std(X)
				df = (X - np.mean(X))/np.std(X)
				st.dataframe(df)
				df.to_csv(path,index=False)

		elif scaling=='MinMax Scaler':
			if st.sidebar.button('Process MinMax Scaler'):
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				df = df.select_dtypes(include=np.number)
				xmin=np.min(df)
				xmax=np.max(df)
				df = (df -xmin) / (xmax -xmin)
				st.dataframe(df)
				df.to_csv(path,index=False)

		elif scaling=='Robust Scaler':
			if st.sidebar.button('Process Robust Scaler'):
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				df = df.select_dtypes(include=np.number)
				q3=df.quantile(0.75)-df.quantile(0.25)
				df =(df - np.median(df))/q3
				st.dataframe(df)
				df.to_csv(path,index=False)


		elif scaling=='MaxAbs Scaler':
			if st.sidebar.button('Process MaxAbs Scaler'):
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				df = df.select_dtypes(include=np.number) 
				df = df /np.max(abs(df))
				st.dataframe(df)
				df.to_csv(path,index=False)
	except:
		pass



exp=st.sidebar.checkbox("Export The File")

if exp:

	try:

		down=["Download as CSV","Download as XLSX"]
		downl=st.sidebar.selectbox("choose the format of the file ",down)

		if downl=='Download as CSV':
			if st.sidebar.button('Process Download as CSV'):
				with open('temp.csv', 'r+') as f:
					st.sidebar.download_button(
					        label="Download Csv",
					        data=f,
					        mime='text/csv',
					        file_name="File.csv",
					        )

		elif downl=='Download as XLSX':
			if st.sidebar.button('Process Download as XLSX'):
				with open('temp.csv', 'r+') as f:
					st.sidebar.download_button(
					        label="Download XLSX",
					        data=f,
					        mime='text/xlsx',
					        file_name="File.xlsx",
					        )
	except:
		pass