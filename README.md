# Python_Expected_Loss_Models

## I. Introduction
### 1. Business Question
<ul>

<li>LendingClub is the largest online loan marketplace, facilitating personal loans, business loans, and financing of medical procedures. Borrowers can easily access lower interest rate loans through a fast online interface.</li>
<li>Like most other lending companies, lending loans to ‘risky’ applicants is the largest source of financial loss (called credit loss). The credit loss is the amount of money lost by the lender when the borrower refuses to pay or runs away with the money owed. In other words, borrowers who defaultcause the largest amount of loss to the lenders. In this case, the customers labelled as 'charged-off' are the 'defaulters'.</li>
<li>If one is able to identify these risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss. Identification of such applicants using EDA and machine learning is the aim of this case study.</li>
<li>In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default. The company can utilise this knowledge for its portfolio and risk assessment.</li>
<li>To develop your understanding of the domain, you are advised to independently research a little about risk analytics (understanding the types of variables and their significance should be enough).</li>

</ul>

<p>In this case study, we will develop a basic understanding of risk analytics in banking and financial services and understand how data is used to minimise the risk of losing money while lending to customers.</p>


### 2. Dataset


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>LoanStatNew</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>id</td>
      <td>ID of Loan.</td>
    </tr>
    <tr>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <td>int_rate</td>
      <td>Interest Rate on the loan.</td>
    </tr>
    <tr>
      <td>grade</td>
      <td>LC assigned loan grade.</td>
    </tr>
    <tr>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER.</td>
    </tr>
    <tr>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <td>issue_d</td>
      <td>The month which the loan was funded.</td>
    </tr>
    <tr>
      <td>loan_status</td>
      <td>Current status of the loan.</td>
    </tr>
    <tr>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened.</td>
    </tr>
    <tr>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers.</td>
    </tr>
    <tr>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies.</td>
    </tr>
    <tr>
      <td>chargeoff_within_12_mths</td>
      <td>Number of charge-offs within 12 months.</td>
    </tr>
  </tbody>
</table>


## II. Process
### 1. Data Handling and Cleaning
<ul>
        <li>Filling NaN values in the dataset and remove NaN</li>
        <li>Remove those duplicate values</li>
        <li>Define type of data</li>
        <li>Check imbalanced</li>
</ul>

### 2. EDA
#### 2.1 Numeric Features:

<img width="800" alt="" src="https://github.com/user-attachments/assets/50c8bc68-d693-4972-b827-817526b42d98">
<img width="800" alt="" src="https://github.com/user-attachments/assets/bc980bfc-759c-4fe4-85c6-e4868444fee6">

From this distribution, we can see:
* loan_amnt: most amount of transactions centralize from <10000 to 30000
* int_rate:  mostly between 5% and 20%, with a tail extending beyond 25%
* annual_inc: annual_inc are mostly below 200000
* dti: dti are mostly below 50, with some values extending up to 1,000

#### 2.2 Category Features:

##### Application_type:

<img width="500" alt="" src="https://github.com/user-attachments/assets/4b8ee450-478e-45bf-9138-235acfed368d">
<img width="800" alt="" src="https://github.com/user-attachments/assets/20f0d6f4-9989-4339-8a9a-8bb540a6544d">

* The majority of loans are from individual applicants, with a higher default rate (13.43%) compared to joint applications (9.72%), though individual loans dominate the dataset.

##### Home_ownership:

<img width="500" alt="" src="https://github.com/user-attachments/assets/063f0318-bb57-42bb-a3d2-5ff649eed3f4">
<img width="800" alt="" src="https://github.com/user-attachments/assets/cf3b05fd-f4ef-4810-b2d0-e51cb139c451">

* Renters have the highest default rate (15.27%), suggesting that home ownership status influences loan default likelihood. Mortgage holders, despite having the largest loan count, show a lower default rate (11.49%), indicating more stability.

##### Purpose:

<img width="500" alt="" src="https://github.com/user-attachments/assets/8583914c-846d-4dc0-9432-def03af9f2d4">
<img width="800" alt="" src="https://github.com/user-attachments/assets/6c292dd2-6231-47af-b042-27c3fed812be">

* The plot shows that "Debt Consolidation" and "Credit Card" purposes have the highest number of non-default loans, with "Small Business" having the highest default percentage at 20.94%, followed by "Educational" at 19.39%.

##### Grade:

<img width="500" alt="" src="https://github.com/user-attachments/assets/b8e74989-c256-4b20-bdf7-3a2bc6a8320d">
<img width="800" alt="" src="https://github.com/user-attachments/assets/1e138a8c-dfdc-472a-9427-a864ef066f16">

* The default rate is highest for grade G at 44.18%, while grade A has the lowest default rate at 3.79%, indicating that lower credit grades have significantly higher default risks.

##### Term:

<img width="500" alt="" src="https://github.com/user-attachments/assets/93f39ac7-5737-447a-8a94-e5866c89a3dc">
<img width="800" alt="" src="https://github.com/user-attachments/assets/f946fd74-13df-4787-97a2-07616c72791f">

* The default rate is higher for loans with a 60-month term at 18.06%, compared to loans with a 36-month term, which have a default rate of 11.08%, indicating that longer-term loans are more likely to default.

##### Annual_income:

<img width="500" alt="" src="https://github.com/user-attachments/assets/d4e09f82-7d43-4cae-bfa6-6d54c132dc0a">
<img width="800" alt="" src="https://github.com/user-attachments/assets/b68e3b57-d4a3-439b-9b14-ce8a89473ef6">

* The trend shows that as income increases, the percentage of defaults decreases. Specifically, the "Basic" income group has the highest default rate (15.48%), while the "Exclusive" income group has the lowest (10.38%). This suggests that higher income is associated with a lower likelihood of default.

### Employee_length:

<img width="500" alt="" src="https://github.com/user-attachments/assets/9b3f8241-2b00-4f30-856e-ace83c7fdac7">
<img width="800" alt="" src="https://github.com/user-attachments/assets/f9b5d003-74a7-403c-af2a-98747b9172b5">

* Employment length shows a similar distribution for defaults and non-defaults, indicating it may not significantly influence loan default rates

## III. Find the Best Model
### Confusion Matrix
#### LogisticRegression

<img width="500" alt="" src="https://github.com/user-attachments/assets/8521b536-6769-413b-ac1d-e5075d2fe2bc">

#### Random Forest Classifier

<img width="500" alt="" src="https://github.com/user-attachments/assets/77d8bdd8-635f-4472-90a4-7b566ea8733e">

#### XGBoot

<img width="500" alt="" src="https://github.com/user-attachments/assets/a9e7960b-d21e-4ad4-80f3-2eb30a386643">

### ROC

<img width="800" alt="" src="https://github.com/user-attachments/assets/6874e30e-6925-41b9-88c3-ff7e54ba6f24">

## III. Calculate Expected Loss

<img width="500" alt="" src="https://github.com/user-attachments/assets/085b9552-ef5e-4fbd-94e1-999cd3c63332">

<img width="500" alt="" src="https://github.com/user-attachments/assets/8a03da32-68aa-45b9-87b8-f7a4cd268693">
