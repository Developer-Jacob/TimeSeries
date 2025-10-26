클린 아키텍처란?

클린 아키텍처는 소프트웨어 설계 원칙과 패턴의 모음으로, 시스템을 유연하고 테스트 가능하며 유지보수하기 쉽게 만드는 것을 목표로 합니다.

클린 아키텍처는 코드의 의존성을 특정 프레임워크, UI, 데이터베이스 등 외부 요소로부터 분리하여 핵심 비즈니스 로직이 독립적으로 동작하도록 합니다.

클린 아키텍처는 주로 다음 원칙에 기반합니다:

단일 책임 원칙(SRP): 각 클래스나 모듈은 단 하나의 책임만 가져야 합니다.

의존성 역전 원칙(DIP): 고수준 모듈이 저수준 모듈에 의존하지 않고, 모두 추상화에 의존해야 합니다.

개방-폐쇄 원칙(OCP): 기존 코드를 수정하지 않고도 기능을 확장할 수 있어야 합니다.

클린 아키텍처의 주요 이점
확장성: 요구사항 변화에 유연하게 대응할 수 있는 구조입니다.

유지보수성 향상: 코드베이스가 잘 구조화되어 있어 새로운 기능을 추가하거나 기존 기능을 수정하기 용이합니다.

테스트 용이성: 각 계층이 독립적으로 동작하므로 단위 테스트와 통합 테스트를 쉽게 작성할 수 있습니다.

의존성 제어: 외부 프레임워크나 라이브러리 의존도를 최소화하여 교체나 업그레이드가 쉬워집니다.

클린 아키텍처의 필요성
1. 복잡성 관리

대규모 프로젝트에서는 기능 추가나 요구사항 변경이 잦습니다. 클린 아키텍처는 코드를 모듈화하여 복잡성을 줄이고 유지보수를 쉽게 만듭니다.

2. 협업 향상

팀 내에서 코드 역할이 명확히 정의되므로 충돌을 최소화하고, 새 팀원이 쉽게 프로젝트에 적응할 수 있습니다.

3. 테스트 개발 지원

독립된 계층 구조는 각 계층의 테스트를 분리하여 작성할 수 있게 하며, 이를 통해 품질 높은 소프트웨어 개발이 가능합니다.

4. 독립성 유지

애플리케이션은 외부 프레임워크나 라이브러리에 의존하지 않도록 설계되므로 기술 스택 변경 시에도 유연하게 대응할 수 있습니다.

클린 아키텍처의 구조
계층별 역할
Presentation Layer(View, ViewModel(StateObject)...)
View


데이터를 사용자에게 표시하고, 사용자로부터 입력을 받아 처리합니다.

비즈니스 로직이 포함되지 않으며, UI 관련 작업만 수행합니다.

StateObject, ViewModel...

UseCase를 호출하여 필요한 데이터를 가져오고, View에서 사용할 수 있도록 상태를 업데이트합니다.
Repository와 직접 상호작용하지 않습니다.
데이터를 가공하고 View에 적합한 형태로 제공합니다.
Domain Layer(Entities, UseCase)

UseCase

역할:

비즈니스 규칙 캡슐화

UseCase는 비즈니스 로직의 중심이며, 애플리케이션의 핵심 규칙과 흐름을 정의합니다.

예를 들어, “사용자 정보를 가져오는 기능”, “주문을 생성하는 기능” 같은 단위 작업을 구현합니다.

계층 간 의사소통 중재

UseCase는 도메인 계층과 다른 계층(Presentation, Data Layer) 간의 중재 역할을 합니다.

Presentation Layer(ViewModel)는 UseCase를 호출하여 데이터를 처리하고, UseCase는 Data Layer를 호출하여 필요한 데이터를 가져옵니다.

유즈케이스별 단일 책임 원칙(SRP) 준수

UseCase는 하나의 특정 작업만 수행하도록 설계됩니다.

예를 들어, FetchUserProfileUseCase는 사용자 프로필을 가져오는 작업만 처리합니다.

의존성 역전 원칙(DIP) 준수

UseCase는 인터페이스(Repository)에 의존하며, 구체적인 구현(예: API 호출, 데이터베이스 조회)은 Data Layer에서 관리합니다.

설명

독립성

UseCase는 Presentation Layer나 Data Layer에 의존하지 않고, 오직 도메인 규칙과 Repository 인터페이스에만 의존합니다.

UseCase는 특정 비즈니스 로직을 캡슐화하여 유지보수를 쉽게 합니다.

로직 수정 시 다른 계층에 영향을 주지 않도록 설계됩니다.

테스트용이성

UseCase는 외부 종속성을 최소화하고, Stub Repository를 사용하여 쉽게 단위 테스트를 작성할 수 있습니다.

재사용성

UseCase는 Presentation Layer(ViewModel, View)에서 호출되어 재사용될 수 있습니다.

여러 ViewModel에서 동일한 UseCase를 호출할 수 있습니다.

상태 관리 없음

UseCase는 상태를 유지하지 않으며, 각 호출에서 필요한 데이터를 처리하고 반환합니다.

Data Layer(Repository, Provider(DataSource))
Repository
역할: 

Repository는 Domain Layer(유즈케이스)와 Data Layer(DataSource) 간의 중재자 역할을 합니다.

비즈니스 로직이 데이터 소스(API, 데이터베이스 등)의 세부 사항에 의존하지 않도록 데이터를 추상화하여 제공합니다.

설명:
Usecase에서 사용할 기능 제고 

UseCase는 데이터 소스의 세부 구현을 알 필요 없이 Repository의 메서드를 호출하여 데이터를 가져옵니다.

예를 들어, fetchUserProfile() 같은 메서드는 UseCase가 직접 데이터베이스 또는 API 호출을 알지 못하도록 캡슐화합니다.

데이터 소스 조합

Repository는 여러 DataSource(SQL 데이터베이스, REST API 등)에서 데이터를 가져와 통합할 수 있습니다.

필요에 따라 캐싱, 로컬-원격 데이터 조합 등의 작업을 처리합니다.

의존성 역전

Repository는 인터페이스를 통해 정의되며, 이 인터페이스를 구현하는 구체적인 클래스는 Data Layer에 존재합니다.

Domain Layer는 Repository 인터페이스에 의존하며, 구현 세부 사항은 Data Layer에 숨겨져 있습니다.

DataSource
역할:

데이터를 실제로 가져오는 구체적인 구현체입니다.

Repository가 비즈니스 로직에 필요한 데이터를 요청하면, DataSource는 외부 데이터 소스(API, 데이터베이스 등)와 통신하여 데이터를 제공합니다.

설명:

구체적인 데이터 처리 책임

REST API 호출, GraphQL 요청, SQLite 쿼리 등 실제 데이터 소스와의 통신 로직을 처리합니다.

외부 데이터 소스의 변경 사항이 Domain Layer에 영향을 미치지 않도록 캡슐화합니다.

단일 책임 원칙 준수

각 DataSource는 특정 데이터 소스와의 상호작용만을 책임집니다.

예를 들어, RemoteDataSource는 원격 API 호출을, LocalDataSource는 데이터베이스 작업만을 처리합니다.

Repository와 협력

Repository는 필요에 따라 여러 DataSource와 협력하여 데이터를 가져오고 조합, 에러처리, 비동기 처리합니다.

즉 데이터를 가져오는 것 이외의 작업은 하지 않는다.

예를 들어, Repository는 먼저 LocalDataSource에서 데이터를 검색하고, 없을 경우 RemoteDataSource에서 가져오는 방식으로 동작할 수 있습니다.

Repository와 DataSource의 관계

Repository는 DataSource에 의존

Repository는 인터페이스를 통해 DataSource를 호출합니다.

UseCase는 특정 데이터 소스의 존재를 알 필요가 없게 됩니다.

DataSource는 구체적인 구현체

RemoteDataSource와 LocalDataSource는 각자의 책임을 수행하며, Repository는 이들을 통합하여 데이터를 제공합니다.


의존성 주입 정책
1. 주입 방식

의존성 주입(DI)은 DI Container를 사용하여 수행한다.
2. Repository 관리 전략

앱의 특성상, 모든 Repository를 동일하게 사용하는 것이 아니라, 일부 Repository의 사용 빈도가 집중될 가능성이 높다.
따라서 초기화 시점에 모든 Repository를 메모리에 로드하는 것은 리소스 낭비로 이어질 가능성이 있으며, 필요 이상으로 앱의 메모리 사용량을 증가시킬 우려가 있다.
3. Lazy Initialization 채택

초기화 시점에 모든 Repository를 생성하는 방식의 이점보다 잠재적 단점(메모리 과부하, 불필요한 초기화 비용)이 더 클 가능성이 크다.
이에 따라, Lazy Initialization 방식을 채택하여 Repository는 필요 시점에서 동적으로 생성 및 주입하도록 설정한다.
4. 중앙 집중 관리

Repository 관리는 한곳에서 통합적으로 이루어지도록 한다.
이러한 방식은 Repository 생명주기 관리의 일관성을 보장하며, 불필요한 중복 초기화를 방지한다.
5. Release 검토

Release 빌드에서 Repository 초기화 전략(Lazy vs. Preload)에 대한 성능 및 메모리 사용 검토는 추후 과제로 설정한다.




Repository를 초기화 시점에 생성 장점
1. 전역적인 사용 범위:

• 대부분의 UseCase나 ViewModel에서 동일한 Repository 인스턴스를 사용해야 하므로, 애플리케이션 전체에서 하나의 인스턴스로 관리하는 것이 효율적이다.

2. 의존성 주입 간소화:

• Repository를 애플리케이션 초기화 시점에 생성하면, 이후 의존성을 주입할 때 필요한 인스턴스를 쉽게 전달할 수 있다.

3. 상태 관리:

• Repository는 캐싱 등 데이터 계층의 상태를 관리할 수 있습니다. 이 상태는 애플리케이션이 실행되는 동안 유지되어야 하므로 전역적으로 관리된다.

4. 중복 방지:

• 초기화 시점에 Repository를 생성하면 여러 UseCase나 ViewModel에서 중복된 인스턴스를 생성할 필요가 없다.



