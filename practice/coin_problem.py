coins = [25, 4,3]
sum = 29
min_coin = min(coins)
max_val = float('inf')
x = sum/min(coins)
count_coins = [max_val] *(sum+1)


for i in range(sum+1):
    coin_count = max_val
    if i == 0:
        count_coins[i] =0
    if i!=0:
        for coin in coins:
            if i >=coin and count_coins[i-coin] != max_val : 

                coin_count = min(coin_count,count_coins[i-coin]+1)
        count_coins[i] = coin_count         
    

print(count_coins)
