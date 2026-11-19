"""CPU-only LLM inference benchmarking toolkit.

Public surface for the published datasets: clients that time a request,
``matrix`` that turns a prompt suite into a campaign, and ``report`` that
turns a campaign back into tables. Nothing here requires a GPU or a cloud
account, and nothing here judges answer quality; speed is the only thing
measured.
"""

__version__ = "0.2.0"
