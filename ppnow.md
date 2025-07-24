
# First question 

## Marking scheme

2 out of 3

---

You have a python dictionary called wallet where:

```python
wallet = {
    "USD": [100, -20, 50, -5],
    "BTC": [0.005, -0.001]
}
```

- **Keys** are currency codes like "USD" or "BTC".
- **Values** are lists of numbers showing past transactions (positive = money in, negative = money out).

## Q1 Add a Transaction
**Goal:** Write a function add_transaction(wallet, currency, amount)

**Do:**
- Append amount to the list for currency.
- If currency isn’t in wallet, create a new list with amount.
- Modify wallet in place (no need to return it).

**Context:**
1. We lost 5 dollars or we added 25 Euro and we would need to record both of them. 
2. If currency is already a key in wallet, add (append) amount to that list.
3. If currency is not in wallet, create a new list with amount as its first item.

The function should change the original dictionary (modify it in place). It does not need to return anything.

```python
wallet = {
    "USD": [100, -20, 50, -5],
    "BTC": [0.005, -0.001],
	"EUR": [25]
}
```

None of them actually loops over the dictionary—each simply checks (or auto-creates) the missing key, then does one append. Choose based on whether you prefer explicit checks, brevity on a plain dict, or an automatic-creation dict type.

### Loop Manual “if” check
```python
def update_wallet(wallet_dict, currency, amount):
    wallet_dict[currency].append(amount)
    return wallet_dict

update_wallet(wallet_dict, "USD", 30)

# But we need both
def update_wallet(wallet_dict, currency, amount):

    if currency not in wallet_dict:
        wallet_dict[currency] = []
    wallet_dict[currency].append(amount) #

    return wallet_dict

update_wallet(wallet_dict, "EUR", 30)>)

```

### `dict.setdefault` one-liner
```python
def add_transaction(wallet: dict, currency: str, amount: float) -> None:
    """
    Update the wallet dict with a new transaction.

    Args:
        wallet: dict like {'USD': [100, -20], 'BTC': [0.005]}
        currency: currency code to update/add
        amount: transaction amount (positive = deposit, negative = withdrawal)

    Effect:
        Modifies `wallet` in place. Creates the currency list if missing.
    """
    wallet.setdefault(currency, []).append(amount)

wallet_dict = {'USD': [100, -20, 50], 'BTC': [0.005, -0.001]}
add_transaction(wallet_dict, 'EUR', 25)
add_transaction(wallet_dict, 'USD', -5)

# wallet becomes:
# {
#   "USD": [100, -20, 50, -5],
#   "BTC": [0.005, -0.001],
#   "EUR": [25]
# }
```

### `collections.defaultdict(list)`

```python

from collections import defaultdict

# Suppose this is your existing wallet dict:
wallet_dict = {
    'USD': [100, -20, 50],
    'BTC': [0.005, -0.001]
}

# Step 1: Convert it into a defaultdict(list)
wallet = defaultdict(list, wallet_dict)

# Now wallet is a defaultdict(list), seeded with all your old data:
print(wallet)
# defaultdict(<class 'list'>, {'USD': [100, -20, 50], 'BTC': [0.005, -0.001]})

# Step 2: Use it exactly like before, but now new currencies auto-create their list:
wallet['EUR'].append(25)
wallet['USD'].append(-5)

print(wallet)
# defaultdict(<class 'list'>, {
#   'USD': [100, -20, 50, -5],
#   'BTC': [0.005, -0.001],
#   'EUR': [25]
# })
```


----

## Q2 Find the sum

Write a function net_balances(wallet) that returns a new dictionary mapping each currency to the **sum** of its transactions.

Now, create a function to calculate the **net balance** of each currency (i.e., the sum of all transactions for that currency).

### Plain for Loop
```python
def net_balances(wallet):
    result = {}
    for currency, txns in wallet.items():
        result[currency] = sum(txns)
    return result
```

### Dict Comprehension
```python
def net_balances(wallet):
    return {c: sum(txns) for c, txns in wallet.items()}

```

### defaultdict
```python
from collections import defaultdict

def net_balances(wallet):
    out = defaultdict(float)
    for c, txns in wallet.items():
        out[c] = sum(txns)
    return dict(out)
```

### map + dict
```python
def net_balances(wallet):
    return dict(map(lambda item: (item[0], sum(item[1])), wallet.items()))
```

### functools.reduce
```python
from functools import reduce

def net_balances(wallet):
    return reduce(lambda acc, kv: (acc.update({kv[0]: sum(kv[1])}) or acc), wallet.items(), {})
```

### Generator Version (if you want an iterator)
```python
def iter_net_balances(wallet):
    for c, txns in wallet.items():
        yield c, sum(txns)

# dict(iter_net_balances(wallet)) to materialize
```

```python
exchange_rates = {
    "USD": 1.0,
    "BTC": 30000.0,  # 1 BTC = 30,000 USD
    "EUR": 1.1       # 1 EUR = 1.1 USD
}
```

---

## Q3 Find the Weight

“Currency weight” = the share of the _total wallet value_ (after converting everything to USD).

Write currency_weights(wallet, rates) that:
1. Uses net_balances to get each currency’s balance.
2. Converts each balance to USD using rates.
3. Returns each currency’s fraction of the total (0–1).
    _(If you prefer percentages, multiply by 100.)_

### Loop
```python
def currency_weights(wallet, rates):
    values = {}
    total = 0.0
    for cur, txns in wallet.items():
        v = sum(txns) * rates.get(cur, 0.0)
        values[cur] = v
        total += v
    if total == 0:
        return {c: 0.0 for c in values}
    return {c: v / total for c, v in values.items()}
```

### Generator
```python
def currency_weights(wallet, rates):
    values = {c: sum(txns) * rates.get(c, 0.0) for c, txns in wallet.items()}
    total = sum(values.values())
    return {c: (v / total if total else 0.0) for c, v in values.items()}
```


```python
def currency_weights(wallet_dict: dict[str, list[float]],
                     rates: dict[str, float]) -> dict[str, float]:
    balances = net_balances(wallet_dict)
    values_usd = {c: balances[c] * rates.get(c, 0.0) for c in balances}
    total = sum(values_usd.values())
    return {c: (v / total if total else 0.0) for c, v in values_usd.items()}
```

### One-pass dict comprehension + generator for total
```python
def currency_weights(wallet, rates):
    values = {c: sum(txns) * rates.get(c, 0.0) for c, txns in wallet.items()}
    total = sum(values.values())
    return {c: (v / total if total else 0.0) for c, v in values.items()}

```

----

**Bonus Questions:**

1. **Data Validation**: How would you handle invalid transactions (e.g., non-numeric values, unsupported currencies)?
2. **Historical Analysis**: Extend the wallet to store timestamps for each transaction. How would you modify your data structure and functions to support querying the balance at a specific point in time?
3. **Visualization**: How would you visualize the portfolio composition over time? What libraries or tools would you use?

---

# Second Question

## Drop rows with missing values and Calculate trade_value
```python
# drop any row that has at least one NaN anywhere
df = df.dropna()

df = df.dropna(subset=['quantity', 'price'])
```

```python
sign = df['side'].str.upper().map({'SELL': -1, 'BUY': 1}).fillna(1)

df['trade_value'] = df['quantity'] * df['price'] * sign

df = df.dropna(axis=1, how='any')  # drop columns that STILL have NaNs (optional)


df = (
    df.assign(trade_value=lambda d: d['quantity'] * d['price'] *
                                   d['side'].str.upper().map({'SELL': -1, 'BUY': 1}).fillna(1))
      .dropna(axis=1, how='any')
)

```
## Net position per ticker
```python
# Net position (shares/contracts) per ticker = BUY qty − SELL qty


# Pivot-table style
pivot = df.pivot_table(index='ticker',
                       columns='side',
                       values='quantity',
                       aggfunc='sum',
                       fill_value=0)
pivot['net_position'] = pivot.get('BUY', 0) - pivot.get('SELL', 0)

# One-liner with groupby
net_pos = df.groupby('ticker').apply(
    lambda d: (d['quantity'] * (d['side'].str.upper().eq('BUY').astype(int)*2 - 1)).sum()
).rename('net_position').reset_index()


sign = df['side'].str.upper().map({'BUY': 1, 'SELL': -1})
net_pos = (
    df.assign(q_signed=df['quantity'].fillna(0) * sign.fillna(0))
      .groupby('ticker', as_index=False)['q_signed'].sum()
      .rename(columns={'q_signed': 'net_position'})
)
print(net_pos)
```

## Total traded value per account
```python
# --- Total traded value per account ---

# If you already have trade_value (qty*price with SELL negative), just do:
total_net = df.groupby('account_id', as_index=False)['trade_value'].sum()

# If not, create it quickly (net = BUY +, SELL -):
sign = df['side'].str.upper().map({'BUY': 1, 'SELL': -1})
df['trade_value'] = df['quantity'] * df['price'] * sign
total_net = df.groupby('account_id', as_index=False)['trade_value'].sum()

# (Optional) Gross traded value (sum of absolute dollars traded):
df['trade_value_gross'] = (df['quantity'] * df['price']).abs()
total_gross = df.groupby('account_id', as_index=False)['trade_value_gross'].sum()

# (Optional) Split BUY/SELL and net in one go:
summary = (
    df.assign(val=df['quantity'] * df['price'])
      .pivot_table(index='account_id',
                   columns=df['side'].str.upper(),
                   values='val',
                   aggfunc='sum',
                   fill_value=0)
)
summary['net'] = summary.get('BUY', 0) - summary.get('SELL', 0)
```


## Bonus Questions[](https://vm-d81e2a0d-abf3-4af3-8b64-4382a8e0a041-ide.us-vmprovider.projects.hrcdn.net/lab/workspaces/1753293130309/tree/finance_trade_analysis_question.ipynb?cloneDefaultIfEmpty=true&sid=dpvnvxtugeqrntfyhvrxhlchprxitphw&qid=2&roleType=datascience&userId=1753293130309&app=interviews&requestStartTimestamp=1753306939779&theme=dark-hrds&e=WyJlbmFibGVRdWVzdGlvbkRlc2NyaXB0aW9uIiwiZW5hYmxlUnVuVGVzdFZpZXciLCJlbmFibGVBaUFzc2lzdGFudENvbGxhYiJd#Bonus-Questions)

1. **Trade Integrity Check**: Identify any trades with negative quantities or prices. How would you handle them?
2. **Time-Based Analysis**: Plot the total daily traded value over time. Are there any noticeable trends or anomalies?
3. **Portfolio Snapshot**: Assume this is end-of-day data. How would you compute the current portfolio value per account, assuming you have a dictionary of latest prices?
