import pandas as pd

# Load data
messages_location = r"c:\Users\lndnc\Downloads\all_chat_data.csv"
contacts_location = r"c:\Users\lndnc\Downloads\contacts.csv"
save = r"c:\Users\lndnc\Downloads\cleanedimessages.csv"

messages = pd.read_csv(messages_location)
contacts = pd.read_csv(contacts_location)

print(messages.shape)
print(messages.columns)
print(contacts.shape)
print(contacts.columns)

def add_plus_prefix(number):
    number = str(number)  
    if number[0].isdigit() and not number.startswith('+'):
        return '+' + number
    return number


contacts['Phone Number'] = contacts['Phone Number'].astype(str).apply(add_plus_prefix)
messages['Sender'] = messages['Sender'].astype(str).apply(add_plus_prefix)

contacts_dict = dict(zip(contacts['Phone Number'], contacts['Full Name']))

messages['Sender'] = messages['Sender'].replace(contacts_dict)
messages['Contact Identifier'] = messages['Contact Identifier'].replace(contacts_dict)
messages.rename(columns={'Contact Identifier': 'To'}, inplace=True)

messages.loc[messages['Group Chat'] == 1, 'To'] = ''

messages

messages.to_csv(save, index=False)

print(messages.head())

for i in range(20):
    print(messages["Message"][i])
    #print(contacts["Phone Number"][i])
    print("---")
