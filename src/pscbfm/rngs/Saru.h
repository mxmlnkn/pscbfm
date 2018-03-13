#pragma once

#include <stdint.h>

#ifdef __CUDACC__
#   define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#   define CUDA_CALLABLE_MEMBER
#endif


class Saru
{
private:
    bool initiliazed;

public:
	using GlobalState = int;

    CUDA_CALLABLE_MEMBER inline Saru();
    CUDA_CALLABLE_MEMBER inline Saru( unsigned int seed );
    CUDA_CALLABLE_MEMBER inline Saru( unsigned int seed1, unsigned int seed2 );
    CUDA_CALLABLE_MEMBER inline Saru( unsigned int seed1, unsigned int seed2, unsigned int seed3 );
    CUDA_CALLABLE_MEMBER inline void init( unsigned int seed1, unsigned int seed2, unsigned int seed3 );

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return false; }

    CUDA_CALLABLE_MEMBER inline void setGlobalState( GlobalState const * ){}
    CUDA_CALLABLE_MEMBER inline void setIteration  ( uint64_t const rIteration ){ iteration = rIteration; }
    CUDA_CALLABLE_MEMBER inline void setSeed       ( uint64_t const rSeed      ){ seed      = rSeed     ; }
    CUDA_CALLABLE_MEMBER inline void setSubsequence( uint64_t const rSubseq    ){ subseq    = rSubseq   ; }
    CUDA_CALLABLE_MEMBER inline uint32_t rng32(void){if( ! initiliazed ) init(seed, iteration, subseq);return u32();}

    //! Run-time computed advancement of the state of the generator
    CUDA_CALLABLE_MEMBER inline void advance(unsigned int steps);

    //! Efficient compile-time advancement of the generator
    /*!
     * \tparam steps Number of steps to advance.
     *
     * The state of the generator is advanced \a steps. This operation is
     * very efficient because it is done through template tricks at compile-time.
     */
    template <unsigned int steps>
    CUDA_CALLABLE_MEMBER inline void advance()
    {
        advanceWeyl<steps>(); advanceLCG<steps>();
    }

    //! Efficient compile-time rewind of the state of the generator.
    /*!
     * \tparam steps Number of steps to advance.
     *
     * The state of the generator is rewound \a steps. This operation is
     * very efficient because it is done through template tricks at compile-time.
     */
    template <unsigned int steps>
    CUDA_CALLABLE_MEMBER inline void rewind()
    {
        /*
         * OK to advance negative steps in LCG, it's done mod 2^32 so it's
         * the same as advancing 2^32-1 steps, which is correct!
         */
        rewindWeyl<steps>(); advanceLCG<-steps>();
    }

    //! Fork the state of the generator
    template <unsigned int seed>
    CUDA_CALLABLE_MEMBER Saru fork() const;

    template< unsigned int steps >
    CUDA_CALLABLE_MEMBER inline unsigned int u32();
    CUDA_CALLABLE_MEMBER inline unsigned int u32();

private:

    uint2 state;    //!< Internal state of the generator

    uint64_t seed;
    uint64_t iteration;
    uint64_t subseq;

    //! \name Internal advancement methods
    //@{
    static const unsigned int LCGA=0x4beb5d59; //!< Full period 32 bit LCG
    static const unsigned int LCGC=0x2600e1f7;
    static const unsigned int oWeylPeriod=0xda879add; //!< Prime period 3666320093
    static const unsigned int oWeylOffset=0x8009d14b;
    static const unsigned int oWeylDelta=(oWeylPeriod-0x80000000)+(oWeylOffset-0x80000000); //!< wraps mod 2^32

    template <unsigned int steps>
    CUDA_CALLABLE_MEMBER inline  void advanceLCG();
    template <unsigned int offset, unsigned int delta, unsigned int modulus, unsigned int steps>
    CUDA_CALLABLE_MEMBER inline unsigned int advanceAnyWeyl(unsigned int x);
    template <unsigned int steps>
    CUDA_CALLABLE_MEMBER inline void advanceWeyl();
    template <unsigned int steps>
    CUDA_CALLABLE_MEMBER inline void rewindWeyl();

    //! \name Advancement helper metaprograms
    //@{
    //! Helper to compute A^N mod 2^32
    template<unsigned int A, unsigned int N>
    struct CTpow
    {
        static const unsigned int value=(N&1?A:1)*CTpow<A*A, N/2>::value;
    };
    //! Template specialization to terminate recursion, A^0 = 1
    template<unsigned int A>
    struct CTpow<A, 0>
    {
        static const unsigned int value=1;
    };

    //! Helper to compute the power series 1+A+A^2+A^3+A^4+A^5..+A^(N-1) mod 2^32
    /*!
     * Based on recursion:
     * \verbatim
     * g(A,n)= (1+A)*g(A*A, n/2);      if n is even
     * g(A,n)= 1+A*(1+A)*g(A*A, n/2);  if n is ODD (since n/2 truncates)
     * \endverbatim
     */
    template<unsigned int A, unsigned int N>
    struct CTpowseries
    {
        static const unsigned int recurse=(1+A)*CTpowseries<A*A, N/2>::value;
        static const unsigned int value=  (N&1) ? 1+A*recurse : recurse;
    };
    template< unsigned int A > struct CTpowseries<A, 0>{ static const unsigned int value=0; };
    template< unsigned int A > struct CTpowseries<A, 1>{ static const unsigned int value=1; };

    //! Helper to compute A*B mod m.  Tricky only because of implicit 2^32 modulus.
    /*!
     * Based on recursion:
     *
     * \verbatim
     * if A is even, then A*B mod m =  (A/2)*(B+B mod m) mod m.
     * if A is odd,  then A*B mod m =  (B+((A/2)*(B+B mod m) mod m)) mod m.
     * \endverbatim
     */
    template <unsigned int A, unsigned int B, unsigned int m>
    struct CTmultmod
    {
        // (A/2)*(B*2) mod m
        static const unsigned int temp=  CTmultmod< A/2, (B>=m-B ? B+B-m : B+B), m>::value;
        static const unsigned int value= A&1 ? ((B>=m-temp) ? B+temp-m: B+temp) : temp;
    };
    //! Template specialization to terminate recursion
    template <unsigned int B, unsigned int m>
    struct CTmultmod<0, B, m>{ static const unsigned int value=0; };
};

/**
 * The default constructor initializes a simple dummy state.
 */
CUDA_CALLABLE_MEMBER inline Saru::Saru()
{
    state.x = 0x12345678;
    state.y = 0x12345678;
    initiliazed = false;
}

/**
 * This seeding was carefully tested for good churning with 1, 2, and 3 bit flips.
 * All 32 incrementing counters (each of the circular shifts) pass the TestU01 Crush tests.
 */
CUDA_CALLABLE_MEMBER inline Saru::Saru(unsigned int seed)
{
    state.x = 0x79dedea3*(seed^(((signed int)seed)>>14));
    state.y = seed ^ (((signed int)state.x)>>8);
    state.x = state.x + (state.y*(state.y^0xdddf97f5));
    state.y = 0xABCB96F7 + (state.y>>1);
    initiliazed=true;
}

/**
 * One bit of entropy is lost by mixing because the input seeds have 64 bits,
 * but after mixing there are only 63 left.
 */
CUDA_CALLABLE_MEMBER inline Saru::Saru(unsigned int seed1, unsigned int seed2)
{
    seed2 += seed1<<16;
    seed1 += seed2<<11;
    seed2 += ((signed int)seed1)>>7;
    seed1 ^= ((signed int)seed2)>>3;
    seed2 *= 0xA5366B4D;
    seed2 ^= seed2>>10;
    seed2 ^= ((signed int)seed2)>>19;
    seed1 += seed2^0x6d2d4e11;

    state.x = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    state.y = (state.x+seed2) ^ (((signed int)state.x)>>8);
    state.x = state.x + (state.y*(state.y^0xdddf97f5));
    state.y = 0xABCB96F7 + (state.y>>1);
    initiliazed=true;
}

CUDA_CALLABLE_MEMBER inline Saru::Saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
{
    init( seed1, seed2, seed3 );
}
/**
 *The seeds are premixed before dropping to 64 bits.
 */
CUDA_CALLABLE_MEMBER inline void Saru::init(unsigned int seed1, unsigned int seed2, unsigned int seed3)
{
    seed3 ^= (seed1<<7)^(seed2>>6);
    seed2 += (seed1>>4)^(seed3>>15);
    seed1 ^= (seed2<<9)+(seed3<<8);
    seed3 ^= 0xA5366B4D*((seed2>>11) ^ (seed1<<1));
    seed2 += 0x72BE1579*((seed1<<4)  ^ (seed3>>16));
    seed1 ^= 0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
    seed2 += seed1*seed3;
    seed1 += seed3 ^ (seed2>>2);
    seed2 ^= ((signed int)seed2)>>17;

    state.x = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    state.y = (state.x+seed2) ^ (((signed int)state.x)>>8);
    state.x = state.x + (state.y*(state.y^0xdddf97f5));
    state.y = 0xABCB96F7 + (state.y>>1);
    initiliazed=true;
}

/*!
 * \param Number of steps to advance the generator.
 *
 * This run-time method is less efficient than compile-time advancement, but is
 * still pretty fast.
 */
CUDA_CALLABLE_MEMBER inline void Saru::advance(unsigned int steps)
{
    // Computes the LCG advancement AND the Weyl D*E mod m simultaneously
    unsigned int currentA=LCGA;
    unsigned int currentC=LCGC;

    unsigned int currentDelta=oWeylDelta;
    unsigned int netDelta=0;

    while (steps)
    {
        if (steps&1)
        {
            state.x=currentA*state.x+currentC; // LCG step
            if (netDelta<oWeylPeriod-currentDelta) netDelta+=currentDelta;
            else netDelta+=currentDelta-oWeylPeriod;
        }

        // Change the LCG to step at twice the rate as before
        currentC+=currentA*currentC;
        currentA*=currentA;

        // Change the Weyl delta to step at 2X rate
        if (currentDelta<oWeylPeriod-currentDelta) currentDelta+=currentDelta;
        else currentDelta+=currentDelta-oWeylPeriod;

        steps/=2;
        }

    // Apply the net delta to the Weyl state.
    if (state.y-oWeylOffset<oWeylPeriod-netDelta) state.y+=netDelta;
    else state.y+=netDelta-oWeylPeriod;
}

/*!
 * \tparam seed Seed for creating the new generator.
 * \returns A new instance of the generator.
 *
 * The user-supplied seed is bitchurned at compile time, which is very efficient.
 * Churning takes small user values like 1 2 3 and hashes them to become roughly
 * uncorrelated.
 */
template <unsigned int seed>
CUDA_CALLABLE_MEMBER Saru Saru::fork() const
{
    const unsigned int churned1=0xDEADBEEF ^ (0x1fc4ce47*(seed^(seed>>13)));
    const unsigned int churned2=0x1234567+(0x82948463*(churned1^(churned1>>20)));
    const unsigned int churned3=0x87654321^(0x87655677*(churned2^(churned2>>16)));

    Saru z;
    z.state.x=churned2+state.x+(churned3^state.y);
    unsigned int add=(z.state.x+churned1)>>1;
    if (z.state.y-oWeylOffset<oWeylPeriod-add) z.state.y+=add;
    else z.state.y+=add-oWeylPeriod;
    return z;
}

/*!
 * \tparam Number of steps to advance.
 * \returns A random 32-bit unsigned integer.
 *
 * \post The state of the generator is advanced \a steps.
 *
 * This method implements the heart of the Saru number generator. The state is
 * advance by \a steps, and is then bitswizzled to return a random integer. This
 * simple generation method has been shown to unconditionally pass a battery of
 * tests of randomness.
 */
template< unsigned int steps >
CUDA_CALLABLE_MEMBER inline unsigned int Saru::u32()
{
    advanceLCG<steps>();
    advanceWeyl<steps>();
    const unsigned int v=(state.x ^ (state.x>>26))+state.y;
    return (v^(v>>20))*0x6957f5a7;
}

CUDA_CALLABLE_MEMBER inline unsigned int Saru::u32(){ return u32<1>(); }

template< unsigned int steps >
CUDA_CALLABLE_MEMBER inline void Saru::advanceLCG()
{
    state.x = CTpow<LCGA,steps>::value*state.x+LCGC*CTpowseries<LCGA,steps>::value;
}

/*!
 * \param x
 *
 * \tparam offset
 * \tparam delta
 * \tparam modulus
 * \tparam steps Number of steps to advance.
 */
template <unsigned int offset, unsigned int delta, unsigned int modulus, unsigned int steps>
CUDA_CALLABLE_MEMBER inline unsigned int Saru::advanceAnyWeyl(unsigned int x)
{
    const unsigned int fullDelta=CTmultmod<delta, steps%modulus, modulus>::value;
    /* the runtime code boils down to this single constant-filled line. */
    return x+((x-offset>modulus-fullDelta) ? fullDelta-modulus : fullDelta);
}

template <unsigned int steps>
CUDA_CALLABLE_MEMBER inline void Saru::advanceWeyl()
{
    state.y = advanceAnyWeyl<oWeylOffset,oWeylDelta,oWeylPeriod,steps>(state.y);
}

template <>
CUDA_CALLABLE_MEMBER inline void Saru::advanceWeyl<1>()
{
    state.y = state.y+oWeylOffset+((((signed int)state.y)>>31)&oWeylPeriod);
}

template <unsigned int steps>
CUDA_CALLABLE_MEMBER inline void Saru::rewindWeyl()
{
    state.y = advanceAnyWeyl<oWeylOffset,oWeylPeriod-oWeylDelta,oWeylPeriod,steps>(state.y);
}


