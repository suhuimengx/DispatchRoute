# coding: utf-8

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.auth.credentials import DerivedCredentials
from huaweicloudsdkiotda.v5.region.iotda_region import IoTDARegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkiotda.v5 import *

if __name__ == "__main__":
    # The AK and SK used for authentication are hard-coded or stored in plaintext, which has great security risks. It is recommended that the AK and SK be stored in ciphertext in configuration files or environment variables and decrypted during use to ensure security.
    # In this example, AK and SK are stored in environment variables for authentication. Before running this example, set environment variables CLOUD_SDK_AK and CLOUD_SDK_SK in the local environment
    ak = "TO9AGL3KUTMENTE520KD"
    sk = "uMt7dxooFDsvQkHm9le1CR9CM9pElX7s2tP1mOvL"

    credentials = BasicCredentials(ak, sk) \
            .with_derived_predicate(DerivedCredentials.get_default_derived_predicate()) \

    client = IoTDAClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(IoTDARegion.value_of("cn-north-4")) \
        .build()

    try:
        request = ListDeviceMessagesRequest()
        response = client.list_device_messages(request)
        print(response)
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)
