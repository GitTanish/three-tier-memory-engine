"""Public MemoryEngine facade.

The engine is the only entry point agents need. It coordinates storage,
retrieval, access control, lifecycle, and context assembly behind one
interface.

Planned operations:

    store(...)        persist a new memory
    retrieve(...)     search and assemble context
    update(...)       correct or supersede a memory
    delete(...)       remove a memory
    share(...)        change ownership or access boundaries
    consolidate(...)  turn episodic records into semantic knowledge

Implementations are added as the underlying contracts settle.
"""
