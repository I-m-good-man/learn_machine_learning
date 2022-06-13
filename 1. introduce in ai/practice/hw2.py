items = [('one', 'two'), ('three', 'four'), ('five', 'six'), ('string', 'a')]

sorted_items = [ ('string', 'a'), ('one', 'two'), ('three', 'four'), ('five', 'six'),]

my_sorted_items = sorted(items, key=lambda x: x[1][-1])

print(my_sorted_items == sorted_items)
