# Загрузка и вывод текста электронного письма
with open('email.txt', 'r') as file:
    email = file.read().replace('\n', '')
print(email)

