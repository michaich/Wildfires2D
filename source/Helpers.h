#pragma once

#include <unordered_map>
#include <sstream>
#include "Operator.h"
#include "Cubism/FluxCorrection.h"

//class used to set the initial conditions
class IC : public Operator
{
  public:
  IC(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  std::string getName() {
    return "IC";
  }
};

/*
 * Buffered file logging with automatic flush.
 *
 * A stream is flushed periodically.
 * (Such that the user doesn't have to manually call flush.)
 *
 * If killing intentionally simulation, don't forget to flush the logger!
 */
class BufferedLogger {
    struct Stream {
        std::stringstream stream;
        int requests_since_last_flush = 0;
        // GN: otherwise icpc complains
        Stream(const Stream& c) {}
        Stream() {}
    };
    typedef std::unordered_map<std::string, Stream> container_type;
    container_type files;

    /*
     * Flush a single stream and reset the counter.
     */
    void flush(container_type::iterator it);
public:

    ~BufferedLogger() {
        flush();
    }

    /*
     * Get or create a string for a given file name.
     *
     * The stream is automatically flushed if accessed
     * many times since last flush.
     */
    std::stringstream& get_stream(const std::string &filename);

    /*
     * Flush all streams.
     */
    inline void flush(void) {
        for (auto it = files.begin(); it != files.end(); ++it)
            flush(it);
    }
};

extern BufferedLogger logger;  // Declared in BufferedLogger.cpp.