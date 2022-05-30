from datetime import date, timedelta

def allsundays(year):
   d = date(year, 1, 1)                    # January 1st
   d += timedelta(days = 6 - d.weekday())  # First Sunday
   while d.year == year:
      yield d
      d += timedelta(days = 7)

prev_month = date(2022, 2, 1).strftime('%B')
curr_month = date(2022, 3, 1).strftime('%B')

for d in allsundays(2022):
   d1 = d.strftime('%d') 
   end = (d + timedelta(days=6))
   d2 = end.strftime('%d') 
   curr_month = end.strftime('%B')
   if curr_month != prev_month:
      print('\n' + curr_month)
      prev_month = curr_month
   print(f'    {d1} – {d2}')
   i guess they must have some consistent heating-bending method for commerical stuff
9:39 AM

'''

January
    02 – 08
    09 – 15
    16 – 22
    23 – 29

February
    30 – 05
    06 – 12
    13 – 19
    20 – 26

March
    27 – 05
    06 – 12
    13 – 19
    20 – 26

April
    27 – 02
    03 – 09
    10 – 16
    17 – 23
    24 – 30

May
    01 – 07
    08 – 14
    15 – 21
    22 – 28

June
    29 – 04
    05 – 11
    12 – 18
    19 – 25

July
    26 – 02
    03 – 09
    10 – 16
    17 – 23
    24 – 30

August
    31 – 06
    07 – 13
    14 – 20
    21 – 27

September
    28 – 03
    04 – 10
    11 – 17
    18 – 24

October
    25 – 01
    02 – 08
    09 – 15
    16 – 22
    23 – 29

November
    30 – 05
    06 – 12
    13 – 19
    20 – 26

December
    27 – 03
    04 – 10
    11 – 17
    18 – 24
    25 – 31
'''