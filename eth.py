import os
from web3.auto import w3

from utils import logger


class Web3Connector():
    """
    currently we support geth only
    """
    rank = 0
    world_size = 1

    def __init__(self, uri):
        os.environ['WEB3_PROVIDER_URI'] = uri
        self.connected = w3.isConnected()
        if self.connected:
            logger.info("connected to node {uri}")

    @property
    def enode(self):
        return w3.geth.admin.nodeInfo if self.connected else None

    @property
    def web3(self):
        return w3.geth.web3 if self.connected else None



if __name__ == "__main__":
    logger.set_log_to_stream()

    URI = "http://172.19.1.95:8645"
    conn = Web3Connector(URI)
    flag = "connected" if conn.connected else "disconnected"
    logger.info(f"web3: {URI} ({flag})")
