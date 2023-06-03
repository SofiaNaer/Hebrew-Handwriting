import random


def check_string(string):
    words = string.split()
    fixed_words = []
    random_number = random.randint(0, 1)
    #final_form_letters = ["ך", "ם", "ף", "ץ"]

    replace_for_mem = ["מ" ,"ב"]
    replace_for_zadik = ["ד","ק"]
    replace_for_nun = ["ו" ,"י"]

    for word in words:

        length = len(word)
        fixed = word

        if(length < 2):
            fixed_words.append(fixed)
            continue

        if fixed[-2:] == 'רנ' or fixed[-2:] == 'חנ' :
            fixed = fixed[:-2] + 'ת'
            length = len(fixed)


        if 'טן' in fixed[:-1]:
            fixed = fixed.replace("טן", "א")
            length  = len(fixed)


        # if 'רן' in fixed[:-1]:
        #     fixed = fixed.replace("רן", "ק")
        #     length = len(fixed)


        if 'ן' in fixed[0:length-1]:
            last_letter = fixed[length-1]
            fixed = fixed[:-1].replace("ן", replace_for_nun[random.randint(0, 1)]) + last_letter


        if 'ך' in fixed[0:length-1]:
            last_letter = fixed[length - 1]
            fixed = fixed[:-1].replace("ך", "ו") + last_letter


        if 'ף' in fixed[0:length-1]:
            last_letter = fixed[length - 1]
            fixed = fixed[:-1].replace("ף" ,"ל")+last_letter


        if 'ץ' in fixed[0:length-1]:
            last_letter = fixed[length - 1]
            fixed = fixed[:-1].replace("ץ", replace_for_zadik[random.randint(0, 1)]) + last_letter

        if 'ם' in fixed[0:length-1]:
            last_letter = fixed[length - 1]
            fixed = fixed[:-1].replace("ם", replace_for_mem[random.randint(0, 1)]) + last_letter


        if fixed[length-1] == 'מ':
            fixed = fixed[::-1].replace('מ', 'םי', 1)[::-1]

        fixed_words.append(fixed)

    result = ' '.join(fixed_words)
    print("after repairing", result)

    return result


# # runner:
# test_string_1 = 'פרויהטןם'
# test_string_2 = ' קוובל גוצבע ובקש אח כט הק'
# test_string_3 = 'רתםאת חעשףה לף הסס'
# test_string_4 = 'שז שץ קתם בשקסח'
# test_string_5 = 'תממה טןחת שוה אלע מיףמ'
# test_string_6 = 'חרגנגןת ףמ עסקמ םמ'
#
# check_string(test_string_1)
# check_string(test_string_2)
# check_string(test_string_3)
# check_string(test_string_4)
# check_string(test_string_5)
# check_string(test_string_6)