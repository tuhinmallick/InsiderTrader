import pandas
import pycountry
import requests


class WorldBankIndicatorsAPI:

    URL = "https://api.worldbank.org/v2/country"

    def _get_country_code(self, country):
        """
        Using `pycountry`, return the ISO 3166-1 alpha-3 country code for corresponding query term.

        See also:
            https://github.com/flyingcircusio/pycountry

        Parameters
        ----------
        country : str

        Returns
        -------
        str
            ISO 3166-1 alpha-3 country code for corresponding query term.

        Raises
        ------
        LookupError
            If the query term is not a valid country.
        """
        return pycountry.countries.search_fuzzy(country)[0].alpha_3

    def _get(self, indicator, country: str = "all", params: dict = {}):
        """
        Retrieve a response, valid JSON response or error, from the World Bank Indicators API.

        See also:
            https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

        Parameters
        ----------
        indicator : str
        country : str, optional
        params : dict, optional

        Returns
        -------
        requests.models.Response
            Return JSON response from the World Bank Indicators API.
        """
        url = f"{self.URL}/{country}/indicator/{indicator}"

        return requests.get(url, params)

    def query(self, indicator, country: list = "all", params: dict = {}):
        """
        Retrieve a response, valid JSON response or error, from the World Bank Indicators API.

        See also:
            https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

        Parameters
        ----------
        indicator : str
            World Bank API Indicator.
        country : list, optional
            List of countries. The country name is converted to ISO 3166-1 alpha-3 country code.
        params : dict, optional
             World Bank API Indicator Query Strings.

        Returns
        -------
        pandas.core.frame.DataFrame
            Return a Pandas DataFrame obtained with response data from World Bank Indicators API.
        """
        if isinstance(country, list):
            country = ";".join([self._get_country_code(c) for c in country])

        params.update({"format": "json", "per_page": 1000})

        response = self._get(indicator, country, params)
        data = response.json()[-1]

        return pandas.json_normalize(data)
