#ifndef __ALL_MIMIC_HEADERS__
#define __ALL_MIMIC_HEADERS__ 1

#include <net/tcp.h>
#include "mimic.h"
#include "protocol/cubic.h"
// #include "protocol/hybla.h"
// #include "protocol/owl.h"
// #include "protocol/bbr.h"
// #include "protocol/vegas.h"

// #define ARM_COUNT 4
// #define INIT_MSG "0:cubic;1:hybla;2:bbr;3:vegas;"

// #define ARM_COUNT 2
// #define INIT_MSG "0:cubic;1:hybla;"


static void doInit(struct sock *sk, struct arm *ca)
{

      cubic_init(sk, ca);

      bictcp_init(sk);

      // hybla_init(sk, protocolId);

    //   bbr_init(sk, protocolId);

      // tcp_vegas_init(sk);

}

static void mimicCongAvoid(struct sock *sk, __u32 ack, __u32 acked, __u32 protocolId){

    // All protocols run at the same time updating their parameters
    // @todo: try to execute sequential for testing but a parallel solution should be implemented
    // only one protocol (protocolID) will modify cwnd or socket related parameters (see inside functions)
    // tcp_vegas_cong_avoid(sk, ack, acked, protocolId);
    bictcp_cong_avoid(sk, ack, acked);
    // hybla_cong_avoid(sk, ack, acked, protocolId);
    // tcp_vegas_cong_avoid(sk, ack, acked, protocolId);

    // @note: BBR has no cong_avoid function



static void doPacketsAcked(struct sock *sk, const struct ack_sample *sample, __u32 protocolId)
{

    // All protocols are executed sequentially (parallel would be best)
    // @todo: overhead?!
    bictcp_acked(sk, sample); // it changes ssthresh internally so protocolID is needed
    // tcp_vegas_pkts_acked(sk, sample); // doesn't change any parameter (protocol ID not needed)

}

// // This is only implemented in bbr (kernel v5.4.0.131)
// static void mimicMain(struct sock *sk, const struct rate_sample *rs, __u32 protocolId){
//     bbr_main(sk, rs, protocolId);
// }

static __u32 mimicSsthresh(struct sock *sk, __u32 protocolId){

    // __u32 bic_ssthresh, reno_ssthresh, bbr_sst, vegas_ssthresh;
    __u32 bic_ssthresh;

    bic_ssthresh = bictcp_recalc_ssthresh(sk);
    // reno_ssthresh = tcp_reno_ssthresh(sk);
    // vegas_ssthresh = tcp_reno_ssthresh(sk);

    return bic_ssthresh;

    // switch(protocolId) {
    //     case 0:
    //         return bic_ssthresh;
    //     case 1:
    //         return reno_ssthresh;
    //     case 2:
    //         return bbr_sst;
    //     case 3:
    //         return vegas_ssthresh;
    //     default:
    //         return bic_ssthresh;
    // }
}


    __u32 reno_cwnd;

    bbr_cwnd = bbr_undo_cwnd(sk);
    reno_cwnd = tcp_reno_undo_cwnd(sk);

    switch (protocolId)
      {
          case 0:
              return reno_cwnd;

          case 1:
              return reno_cwnd;

          case 2:
              return bbr_cwnd;

          case 3:
              return reno_cwnd;

          default:
              return reno_cwnd;
     }

}

static void mimicCongestionEvent(struct sock *sk, enum tcp_ca_event event, __u32 protocolId){
    // Only reading operation on socket structure
    // All functions must be executed as they're not returning any value.
    // Functions internally modify their internal variables defined in the global mimic structure in `mimic.h`.
    // Only protocols that support a congestion event are executed (e.g., hybla does not have any cong_event function)

    bictcp_cwnd_event(sk, event);
    bbr_cwnd_event(sk, event, protocolId);
    tcp_vegas_cwnd_event(sk, event);

    // switch (protocolId)
    //   {
    //       case 0:
    //           return bictcp_cwnd_event(sk, event, ca);

    //       case 1:
    //           return;

    //       case 2:
    //           return bbr_cwnd_event(sk, event, ca);

    //       case 3:
    //           return tcp_vegas_cwnd_event(sk, event, ca);

    //       default:
    //           return bictcp_cwnd_event(sk, event, ca);

    //  }
}

static void mimicState(struct sock *sk, __u8 new_state){
    // This function is executed on cwnd event (see tcp_congestion_ops struct)

    bictcp_state(sk, new_state);
    bbr_set_state(sk, new_state);
    tcp_vegas_state(sk, new_state);
    hybla_state(sk, new_state);

    // switch (protocolId)
    //   {
    //       case 0:
    //           return bictcp_cwnd_event(sk, event, ca);

    //       case 1:
    //           return ;

    //       case 2:
    //           return bbr_cwnd_event(sk, event, ca);

    //       case 3:
    //           return;

    //       default:
    //           return bictcp_cwnd_event(sk, event);

    //  }
}

static __u32 mimicSndbufExpand(struct sock *sk){
    return bbr_sndbuf_expand(sk);
}

#endif
