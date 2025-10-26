

✅ Phase 2 목표

UI, Domain, Data 계층을 논리적으로 분리한 상태에서 개발 진행 중

현재는 모든 계층이 하나의 Target(Core) 내에 포함되어 있음

의존성 정리 및 물리적 분리를 통해 구조적 안정성과 유연성 확보를 목표로 함




🧩 현재 구조 이슈
1. 외부 의존성 노출

외부 모듈 및 SDK 의존성이 UI, Domain, Data 전체에 노출됨

ViewModel, UseCase 내부에서 외부 라이브러리 직접 사용되는 사례 존재

2. Feature 간 강결합

Feature 간 의존성 높음

특정 Feature가 외부 SDK에 강하게 결합되어 있었고, 해당 SDK가 SwiftUI #Preview 미지원

이 Feature를 여러 Feature가 참조함 → Preview 기능 전체 사용 불가 상태 발생함




📌 문제 요약

논리적 계층은 분리되어 있으나, 타겟이 분리되지 않아 모든 의존성이 물리적으로 연결되어 있음

결과적으로 외부 변경이나 제약사항이 전체 프로젝트에 영향을 미침




🎯 Phase 2 진행 방향

각 Feature 단위로 타겟 분리, 불필요한 의존성 전파 방지

외부 SDK는 Interface 계층을 통해 주입 → 실제 구현은 Data에서만 접근 가능하게 조정

Preview 지원 확보, 유닛 테스트 작성 용이성 증가, 빌드 안정성 향상




🧭 구조 방향성

클린 아키텍처(Clean Architecture)를 철학적 기반으로 삼되

실용성과 생산성을 고려하여 완전한 계층 분리보다는 유지보수와 테스트 효율을 우선한 구조로 조정함




🎯 설계 목적

물리적 분리를 통한 의존성 명확화

Swift Package Manager(SPM) 기반의 독립된 타겟 구성으로

의존성 전파 차단, 빌드 시간 단축, 모듈 간 영향도 최소화




🧩 UI와 Domain의 통합

ViewModel에서 도메인 로직(UseCase)을 직접 호출하는 구조로 설계

기능 단위 Feature에서는 계층 분리를 줄여

불필요한 추상화를 피하고 코드량과 복잡도를 완화함




📦 Data 계층의 완전한 분리

외부 의존성(API, Storage, SDK 등)은 Data 계층에만 존재

UI 또는 Domain에서는 외부 라이브러리에 접근하지 않도록 격리

문제 발생 가능성이 높은 의존성은 최소한의 경로를 통해 주입됨




🛠️ UseCaseFactory 도입

ViewModel에서 UseCase를 직접 생성하지 않고

UseCaseFactory를 통해 주입받는 구조로 변경

의존성 생성 책임을 일원화하고, 테스트 시 유연하게 교체 가능하도록 구성




👀 프리뷰 및 테스트 친화 구조

Data 계층은 언제든지 Mock 또는 Stub으로 교체 가능

ViewModel이 외부 의존성 없이 초기화될 수 있어

SwiftUI Preview와 Unit Test 작성이 쉬움




구조




이상적인 구조

클린 아키텍처를 기반으로 하나의 Feature 내부를 다음과 같이 4개 계층으로 분리한 모습:

Presentation → Interface ← Domain ← Data

Feature 내부 구조

📌 특징

ViewModel은 Interface에만 의존 → 테스트 용이

Data는 Domain의 Protocol만 알 뿐, 상위 계층에 의존하지 않음

완전한 의존성 역전 구조 유지 (의존성 방향은 Data → Domain → Interface <- Presentation)

Feature 간 구조

🎯 구조 요약

Feature 간 연결을 인터페이스 중심으로 구성한 모습.

각 Feature는 자신의 Interface 모듈을 외부에 제공하고, 다른 Feature는 이 Interface에만 의존함.

✅ 계층별 설명

Feature A, B: 각 Feature는 Interface와 ETC 모듈(UseCase, ViewModel, ... 등)로 구성

CommonUI, Foundation: 공통 UI 구성요소 및 기반 유틸리티를 하위 계층으로 배치

📌 특징

Feature 간 강결합 대신 Interface 간 연결로 설계

확장성 있는 구조이며, Feature 수가 늘어나도 의존성 복잡도 증가 억제 가능
샘플 앱 구성 구조

🎯 구조 요약

앱이 어떤 방식으로 Feature 및 Interface에 의존하고,

외부 Feature는 Stub 또는 Mock으로 대체하는 예시를 보여줌

✅ 구성 설명

Sample App A: PresentationA, InterfaceA, FeatureA를 조합해 구성됨

Interface B, Stub(Feature B): Sample App A는 실제 Feature B를 사용하지 않고, 해당 Interface와 Stub만 참조함

테스트 환경, SampleApp 만들기 적합, QA, 기획 comm용이

📌 특징

앱이 직접 Feature 구현체를 의존하지 않고 Interface를 통해 느슨하게 연결

특정 Feature를 Stub으로 대체 가능하여 SampleApp, 테스트 유연성 확보




https://docs.tuist.dev/en/guides/develop/projects/tma-architecture

Feature 내부의 의존성

🎯 구조 요약

Feature는 Presentation과 Domain 계층을 하나의 모듈로 구성하고, 별도의 Data 모듈을 통해 실제 의존성을 주입받는 구조를 가지고 있음.

클린 아키텍처 원칙을 기반으로 하되, 계층 구분은 실용적인 수준으로 단순화되어 있음.

✅ 계층별 설명
Presentation: View, Presenter를 포함하며 사용자 인터랙션과 상태 처리를 담당
Domain: UseCase와 RepositoryProtocol, Entities를 포함한 핵심 비즈니스 로직 계층
Data: Repository와 Providable, Provider 등 외부 의존성 구현체로 구성됨
📌 특징
ViewModel은 UseCase를 직접 호출하고, UseCase는 RepositoryProtocol을 통해 추상화된 방식으로 Repository에 의존
Data 모듈은 Repository를 통해 Domain의 인터페이스를 구현하며, 외부 API 또는 시스템 접근은 Provider 계층에 집중
테스트와 Preview를 위해 Data 계층만 교체 가능한 구조로 설계됨

Feature 간 의존성

Feature 간 의존성
🎯 구조 요약

각 Feature는 기능 단위로 독립되어 있으나, 일부 기능 공유 및 구현 계층에서 직접 참조가 발생하고 있음.

공통 컴포넌트는 CommonUI, Foundation 모듈로 분리되어 있으며, 모든 Feature의 기반으로 작동함.

✅ 구성 설명
Feature A, Feature B는 각각 Presentation과 Domain 계층을 포함하고, 각자의 Data 모듈을 별도로 가짐
Feature 간에는 직접 참조가 존재하며, 특정 Feature의 Data 계층이 외부 SDK(ApolloNuguKit)와 강하게 연결되어 있음
공통 요소는 CommonUI → Foundation 순으로 공유 구조 하단에 위치
📌 특징
교차 참조 구조로 인해 일부 Feature 변경 시 연쇄적인 영향이 발생할 수 있음
외부 의존성과 강결합된 Feature가 전체 안정성에 영향을 줄 수 있는 구조
구조 개선을 위해 Interface 분리 또는 Stub/Mock 적용을 통한 분리 필요




let package = Package(
    name: "ApolloSleep",
    defaultLocalization: "kr",
    platforms: [.iOS(.v16)],
    products: [
        .library(
            name: "ApolloSleep",
            targets: ["ApolloSleep"]
        ),
        .library(
            name: "ApolloSleepData",
            targets: ["ApolloSleepData"]
        )
    ],
    dependencies: [
        .package(path: "../ApolloFoundation"),
        .package(path: "../ApolloCommonUI"),
        .package(path: "../ApolloNuguKit"),
        .package(path: "../ApolloWebView"),
        .package(path: "../ApolloLogin"),
        .package(url: "https://github.com/asleep-ai/asleep-sdk-ios.git", exact: "2.4.8")
    ],
    targets: [
        .target(
            name: "ApolloSleep",
            dependencies: [
                "ApolloFoundation",
                "ApolloCommonUI",
                "ApolloWebView",
                "ApolloLogin"
            ],
            path: "ApolloSleep",
            sources: ["Sources"],
            resources: [.process("Resources")],
            swiftSettings: [
                .define("BETA", .when(configuration: .debug)),
                .define("PRODUCT", .when(configuration: .release))
            ]
        ),
        .target(
            name: "ApolloSleepData",
            dependencies: [
                "ApolloSleep",
                "ApolloFoundation",
                "ApolloNuguKit",
                .product(name: "AsleepSDK", package: "asleep-sdk-ios")
            ],
            path: "ApolloSleepData",
            sources: ["Sources"],
            swiftSettings: [
                .define("BETA", .when(configuration: .debug)),
                .define("PRODUCT", .when(configuration: .release))
            ]
        )
    ]
)




 













