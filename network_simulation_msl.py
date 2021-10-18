import netsquid as ns
from netsquid.qubits.qformalism import QFormalism
from netsquid.examples.network_simulation import EGService, NetworkProtocol, create_example_network
from netsquid.qubits import StateSampler
from read_network_function import read_graph
import time

surfnet_graph = "Surfnet.graphml.xml"
uscarrier_graph = "UsCarrier.graphml"


class MagicEGProtocol(EGService):
    """Extends the EGService by magically creating the qubits.

    Skips communicating with the other service and 'magically' puts qubits
    into the node's memory. The service sends a response signal once the protocol is finished.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node`
        The node this protocol runs on.
    name : str, optional
        The name of this protocol. Default 'MagicEGProtocol'.

    Attributes
    ----------
    req_create : namedtuple
        A request to create entanglement with a remote node.
    res_ok : namedtuple
        A response to indicate a create request has finished.

    """

    def __init__(self, node, name="MagicEGProtocol"):
        super().__init__(node, name=name)
        # Sample our expected states weighted by their probabilities
        self._sampler = StateSampler([ns.b01, ns.b11], [0.5, 0.5])
        self._other_service = None

    def add_other_service(self, service):
        self._other_service = service

    def handle_request(self, request, identifier, start_time=None, **kwargs):
        """Schedule the request.

        Schedule the request in a queue and
        signal to :meth:`~netsquid.examples.simple_link.EGProtocol.run`
        new items have been put into the queue.

        Parameters
        ----------
        request :
            The object representing the request.
        identifier : str
            The identifier for this request.
        start_time : float, optional
            The time after which the request can be executed.
        kwargs : dict
            Additional arguments not part of the original request.

        Returns
        -------
        dict
            The dictionary with additional arguments.
            For the create request this is the unique entanglement id.

        """
        if start_time is None:
            # By default we wait until we expect the message to have arrived
            travel_time = 10000
            start_time = ns.sim_time() + travel_time
        if kwargs.get('create_id') is None:
            kwargs['create_id'] = self._get_next_create_id()
        self.queue.append((start_time, (identifier, request, kwargs)))
        # Send ourselves a signal we got a new request
        self.send_signal(self._new_req_signal)
        return kwargs

    def create(self, purpose_id, number, create_id, **kwargs):
        """Magic handler of create requests.

        Parameters
        ----------
        purpose_id : int
            The number used to tag this request for a specific purpose in a higher layer.
        number : int
            The number of qubits to make in this request.
        create_id : int
            The unique number associated with this request.

        Yields
        ------
        :class:`~pydynaa.core.EventExpression`
            The expressions required to execute the create request.

        """
        for curpairs in range(number):
            # We wait a bit, this should be sampled from a distribution.
            yield self.await_timer(1000000)
            # We reserve position 0 for the communication qubit
            qpos = curpairs + 1
            qubits = ns.qubits.create_qubits(2)
            qstate, _, _ = self._sampler.sample()
            ns.qubits.assign_qstate(qubits, qstate)
            self.node.qmemory.put(qubits[0], positions=[qpos])
            # Here we cheat by putting the qubits into the memory manually
            self._other_service.node.qmemory.put(qubits[1], positions=[qpos])
            response = self.res_ok(purpose_id, create_id, qpos)
            self.send_response(response)
            # Cheat again to make sure the response is also send via Bob
            self._other_service.send_response(response)


def setup_protocol(network, req_num, node0, node1):
    """Configure the protocols.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        The network to configure the protocols on. Should consist of two nodes
        called Alice and Bob.

    Returns
    -------
    :class:`~netsquid.protocols.protocol.Protocol`
        A protocol describing the complete simple link setup.

    """
    nodes = network.nodes
    alice_egp = MagicEGProtocol(nodes[node0])
    bob_egp = MagicEGProtocol(nodes[node1])
    # The services need a reference to each other in order to cheat
    alice_egp.add_other_service(bob_egp)
    bob_egp.add_other_service(alice_egp)
    return NetworkProtocol("SimpleLinkProtocol 0 1", alice_egp, bob_egp, req_num, node0, node1)


def run_simulation():
    """Run the example simulation.

    """
    ns.sim_reset()
    ns.set_random_state(42)  # Set the seed so we get the same outcome
    ns.set_qstate_formalism(QFormalism.DM)
    (DG,UG) = read_graph(surfnet_graph) # DG: directed graph, UG: undirected graph
    #print (DG, UG)
    #create quantum network
    network = create_example_network(UG)
    # check value of created node
    #for node in network.nodes.values():
        #print(node.name, node.qmemory)
    i = 1
    for edge in UG.edges:
        node_name0 = str(edge[0])
        node_name1 = str(edge[1])
        protocol = setup_protocol(network, i, node_name0, node_name1)
        protocol.start()
        ns.sim_run()
        i = i + 1


if __name__ == "__main__":
    run_simulation()
