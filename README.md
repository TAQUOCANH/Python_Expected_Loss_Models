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
#### Numeric Features:

<img width="800" alt="" src="https://github.com/user-attachments/assets/50c8bc68-d693-4972-b827-817526b42d98">
<img width="800" alt="" src="https://github.com/user-attachments/assets/bc980bfc-759c-4fe4-85c6-e4868444fee6">
