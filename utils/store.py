from utils.influxdb.connection import HistoricalConnection
from collections import deque
import settings


class Store:
    def __init__(self, connection=settings.INFLUXDB_CONNECTION, time_scales=None):
        assert isinstance(time_scales, list)

        self.connection = HistoricalConnection(**connection)
        self.time_scales = time_scales
        self.df = deque(maxlen=len(self.time_scales))

    def get_all_data(self):
        data = self.connection .get_data(start_date=START_DATE, end_date=END_DATE, symbol='ES', type='Min', frame=1)
        print(list(self.connection .get_first_cons(measurement='ohlc', start=START_DATE, end=END_DATE, symbol='ES', type='Min', frame=1))[0])