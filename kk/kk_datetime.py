# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime


# 判断是不是闰年
def year_isleap(year):
    if year % 400 == 0:
        return True
    elif year % 4 == 0 and year % 100 != 0:
        return True
    else:
        return False


def year_month_jieqi(year, month, jie0qi1):
    jieqi = 0
    if month > 1:
        jieqi = (month-2) + jie0qi1
    else:
        jieqi = 22 + jie0qi1
    return year_jieqi(year, jieqi)


def year_jieqi(year, jieqi):
    if (year < 1900) or (year > 2100):
        # 时间错误
        return datetime.date(1900, 1, 1)
    # 从20世纪开始，每年的立春日都是2月4日或2月5日 (20 世纪 4.15)
    _Cs = [[4.6295, 3.87],      # 立春 0
           [19.4599, 18.73],    # 雨水    2026年的计算结果减1日
           [6.3826, 5.63],      # 惊蛰
           [21.4155, 20.646],   # 春分    2084年的计算结果加1日
           [5.59, 4.81],        # 清明
           [20.888, 20.1],      # 谷雨
           [6.318, 5.52],       # 立夏 6  1911年的计算结果加1日
           [21.86, 21.04],      # 小满    2008年的计算结果加1日
           [6.5, 5.678],        # 芒种    1902年的计算结果加1日
           [22.20, 21.37],      # 夏至    1928年的计算结果加1日
           [7.928, 7.108],      # 小暑    1925年和2016年的计算结果加1日
           [23.65, 22.83],      # 大暑    1922年的计算结果加1日
           [28.35, 7.5],        # 立秋 12 2002年的计算结果加1日   # 8.318, 7.5
           [23.95, 23.13],      # 处暑
           [8.44, 7.646],       # 白露    1927年的计算结果加1日
           [23.822, 23.042],    # 秋分    1942年的计算结果加1日
           [9.098, 8.318],      # 寒露
           [24.218, 23.438],    # 霜降    2089年的计算结果加1日
           [8.218, 7.438],      # 立冬 18 2089年的计算结果加1日
           [22.60, 21.94],      # 小雪    1978年的计算结果加1日 # 23.08, 22.36
           [7.9, 7.18],         # 大雪    1954年的计算结果加1日
           [22.36, 21.94],      # 冬至     # 1918年和2021年的计算结果减1日  # 22.6, 21.94
           [6.11, 5.4055],      # 小寒     # 1982年计算结果加1日，2019年减1日  # 7.646, 6.926
           [20.84, 20.12]]      # 大寒 23  # 2000年和2082年的计算结果加1日  21.94, 21.37

    _year00R = year // 100  # 取整 year 的前 2位
    _year00L = year % 100   # 取余 year 的后 2位

    # (Y * D + C ] - L
    _L = int(_year00L * 0.2422 + _Cs[jieqi][_year00R - 19]) - (_year00L - 1) // 4
    if (jieqi == 1 and year == 2026) or \
            (jieqi == 21 and (year == 1918 or year == 2021)) or \
            (jieqi == 22 and year == 2019):
        _L = _L - 1
    elif (jieqi == 3 and year == 2084) or \
            (jieqi == 6 and year == 1911) or \
            (jieqi == 7 and year == 2008) or \
            (jieqi == 8 and year == 1902) or \
            (jieqi == 9 and year == 1928) or \
            (jieqi == 10 and (year == 1925 or year == 2016)) or \
            (jieqi == 11 and year == 1922) or \
            (jieqi == 12 and year == 2002) or \
            (jieqi == 14 and year == 1927) or \
            (jieqi == 15 and year == 1942) or \
            (jieqi == 17 and year == 2089) or \
            (jieqi == 18 and year == 2089) or \
            (jieqi == 19 and year == 1978) or \
            (jieqi == 20 and year == 1954) or \
            (jieqi == 22 and year == 1982) or \
            (jieqi == 23 and (year == 2000 or year == 2082)):
        _L = _L + 1

    _M = jieqi // 2 + 2 if jieqi < 22 else 1
    # 返回一个日期值
    _date = datetime.date(year, _M, _L)
    return _date

# 只能计算 1900 - 2100年之间的日期
def date_ganzhi(year, month, day, hour, out=0):
    if out > 0:
        _gans = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        _zhis = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    else:
        _gans = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        _zhis = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']


    # 年
    _nian_gan = _gans[(year - 4) % 10]
    _nian_zhi = _zhis[(year - 4) % 12]

    # 月
    _yue_gan = _gans[((year - 4) % 5 * 12 + month) % 10]
    # 获取当前的节，
    date = year_month_jieqi(year, month, 0)
    if day > date.day:
        _yue_zhi = _zhis[month - 1]
    else:
        _yue_zhi = _zhis[(month + 12 - 1 )% 12]

    # 日
    year_l00 = year // 100
    year_r00 = year % 100
    _day_gan_N = (year_l00 + year_r00) + year_l00 // 4 + year_r00 // 4
    _day_gan_N = _day_gan_N + ((month + 12 if month < 3 else month) + 1) * 3 // 5
    _day_gan_N = _day_gan_N + day - 3 - year_l00
    _day_gan = _gans[_day_gan_N % 10]

    # 日
    # 1900——1999年 日干支基数 =（年尾二位数+3）*5+55+（年尾二位数-1）除4
    # 2000——2099年 日干支基数=（年尾二位数+7）*5+15+（年尾二位数+19）除4（只用商数，余数不用，数过60就去掉60）
    _day_gan_N = 0
    if year < 2000:
        _day_gan_N = (year % 100 + 3) * 5 + 55 + (year % 100 - 1) // 4
    else:
        _day_gan_N = (year % 100 + 7) * 5 + 15 + (year % 100 + 19) // 4
    _day_gan_N = _day_gan_N % 60

    __days = [31, 29 if year_isleap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(month - 1):
        _day_gan_N = _day_gan_N + __days[i]
    _day_gan_N = _day_gan_N + day
    _day_gan_N = _day_gan_N - 1

    _day_gan = _gans[_day_gan_N % 10]
    _day_zhi = _zhis[_day_gan_N % 12]

    # 时
    # 时支公式：时支=小时÷2-1（小时为偶数），时支=（小时+1）÷2-1（小时为奇数）
    _hour_zhi = ((hour + 1) // 2) % 12
    # 时干公式：时干=日干×2+时支（晨子=-1，夜子=11），（如果和大于10，则取个位数；如果和为20，则取10）
    _hour_gan = (_day_gan_N % 10 * 2 + _hour_zhi + 10) % 10

    _hour_gan = _gans[_hour_gan]
    _hour_zhi = _zhis[_hour_zhi]

    return _nian_gan, _nian_zhi, _yue_gan, _yue_zhi, _day_gan, _day_zhi, _hour_gan, _hour_zhi


'''
天干地支怎么算?
天干地支是四柱八字的根本，为了方便记忆其留下来很多的口诀，下面为大家介绍十天干十二地支口诀表。
十天干十二地支口诀表
命理预测学是自然五行旺弱变化当中寻找兴衰的过程，十天干十二地支是五行的演变，也是预测学中不可缺少的基础知识。
天干地支
1、地支
子—鼠、丑—牛、寅—虎、卯—兔。
辰—龙、巳—蛇、午—马、未—羊。
申—猴、酉—鸡、戌—狗、亥—猪。
子丑寅卯辰巳午未申酉戌亥，子丑因猫沉思无谓神游四海。
2、天干
甲乙丙丁戊己庚辛壬癸，甲乙丙丁无计更新人亏。
天干地支计算方法
1、年干支计算公元后年份的口诀
公元年数先减三，除10余数是天干，基数改用12除，余数便是地支年。
2、月干支月的地支
固定的如正月起寅之类，只计算月干。月干=年干数乘2+月份例：2010年（庚寅）三月（辰月）的天干=7*2+3=17,天干10为周期，就去掉10，得7，天干第7位为庚，则此月干支为庚辰。
3、日干支
1900——1999年 日干支基数 =（年尾二位数+3）*5+55+（年尾二位数-1）除4
2000——2099年 日干支基数=（年尾二位数+7）*5+15+（年尾二位数+19）除4（只用商数，余数不用，数过60就去掉60）
例：2010年4月12日星期一  日干支基数 =（10+7）* 5 + 15 +（10+19）/4=47（已去掉60的倍数）这就是1月1日的干支数。
从1月1日到4月12日为 47（日干支基数）+31（1月天数，下类推）+28+31+12=149，去掉60的倍数得29、天干去10的倍数余9为壬，地支去12的倍数余5为辰，今天的干支就是壬辰。
4、时干支时干
日干序数*2+日支序数-2
五。倒推年龄法从今年的干支推出任何年龄的干支，即年龄去掉60为基数，去掉10的倍数为天干倒推数，去掉12的倍数为地支倒推数。
如今年为庚寅年，56岁生年的干支这样推：56去50余6，天干从庚倒推6位是乙，地支为56去掉48余8，从寅倒推8位是未，生年就是乙未年。
65岁生年去掉60余5，从今年的天干倒推5位丙，从今年的地支倒推5位戌，生年就是丙戌年。

推算时辰的干支要用公式计算，具体如下：
60时辰合5日一个周期；一个周期完了重复使用，周而复始，循环下去。必须注意的是子时分为0时到1时的早子时（晨子）和23时到24时的晚子时（夜子），所以遇到甲或己之日，0时到1时是甲子时，但23时到24时是丙子时。晚子时又称子夜或夜子。
日上起时亦有歌诀：甲己还加甲，乙庚丙作初；丙辛从戊起，丁壬庚子居；戊癸何方发，壬子是真途。
下表列出日天干和时辰地支构成的时辰干支，以北京时间（UTC+8）为准：

'''


def date_ganzhi2(year, month, day, hour):
    _gans = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬','癸']
    _zhis = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未','申', '酉', '戌', '亥']
    _shengxiao = ['鼠', '牛', '虎', '兔', '龙', '蛇','马', '羊', '猴', '鸡', '狗', '猪']
    _yuefen = ['十一', '十二', '正', '二', '三', '四','五', '六', '七', '八', '九', '十']
    _riqi = ['初一', '初二', '初三', '初四', '初五', '初六','初七', '初八', '初九', '初十', '十一', '十二', '十三', '十四','十五', '十六', '十七', '十八', '十九', '二十','廿一', '廿二', '廿三', '廿四', '廿五', '廿六','廿七', '廿八', '廿九', '三十']
    _nian = year - 1900 + 36
    _yue = month - 1
    _ri = day - 1
    _gan = _gans[_nian % 10]
    _zhi = _zhis[_nian % 12]
    _sx = _shengxiao[_nian % 12]
    _yf = _yuefen[_yue]
    _rq = _riqi[_ri]

    # 时

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    year = 2010
    t = (year % 100 + 7) * 5 + 15 + (year % 100 + 19) // 4
    print(t)

    print(date_ganzhi(2023, 10, 26, 5))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
