export class GeneralUtil {
    static numberFormatter = number => {
      if (number < 10) {
        return `0${number}`;
      }
      return number;
    };
    static months = month => {
      const months = [
        'Jan',
        'Feb',
        'Mar',
        'April',
        'May',
        'June',
        'July',
        'Aug',
        'Sept',
        'Nov',
        'Dec',
      ];
      return months[month];
    };
    static datetimeFormatter = (datetime, dateOnly) => {
      if (!datetime) {
        alert('Provide some date and time!');
      }
      const date = new Date(datetime);
      const currentMonth = this.months(
        date.getMonth() > 9 ? date.getMonth() - 1 : date.getMonth(),
      );
      const currentDate = this.numberFormatter(date.getDate());
      const currentYear = date.getFullYear();
      let currentHour = this.numberFormatter(date.getHours());
      const currentMinutes = this.numberFormatter(date.getMinutes());
      const currentDayNight = currentHour > 12 && currentHour < 24 ? 'PM' : 'AM';
      currentHour =
        currentHour > 12 && currentHour < 24 ? currentHour - 12 : currentHour;
      return dateOnly
        ? `${currentMonth} ${currentDate}, ${currentYear}`
        : `${currentMonth} ${currentDate}, ${currentYear} at ${currentHour}:${currentMinutes} ${currentDayNight}`;
    };
  
    static transactionStatus = status => {
      if (status) {
        return GeneralUtil.datetimeFormatter(status?.toDate(), true);
      }
      return 'Pending';
    };
  
    static userDataHandler = obj => {};
  }